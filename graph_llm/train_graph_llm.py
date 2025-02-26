import json
import pathlib
import pickle
import transformers
import torch
import os
import copy
import gc
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import default_data_collator
import os
from dataset import GraphLLMDataset
from model.model import Transformer
from model.model_args import ModelArgs
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
import wandb
from utils import set_seed, adjust_lr
from config import parse_args_llama, task_level

from transformers import LlamaTokenizer

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)


def main(args, seed, accelerator):
    group = f"{args.dataset}"
    accelerator.init_trackers(
        project_name=f"{args.project}",
        init_kwargs={
            "wandb": {
                "tags": [args.dataset, args.model_name],
                "group": group,
                "name": f"{args.dataset}_EXP{seed}",
                "config": args,
            }
        },
    )

    set_seed(seed)
    accelerator.print(args)

    with accelerator.main_process_first():
        # TODO: change to deepseek
        tokenizer = LlamaTokenizer.from_pretrained("Llama-2-7b-hf")
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"

        graph_dataset = GraphLLMDataset(args.module_path, args.dataset, tokenizer)
        dataset, split, edge_index = graph_dataset.get_dataset()
        original_dataset = dataset.map(
            graph_dataset.preprocess_function,
            batched=True,
            batch_size=None,
            remove_columns=[i for i in dataset.column_names if i not in ["node_ids"]],
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=1,
        ).with_format("torch")
        clm_dataset_train = dataset.map(
            graph_dataset.preprocess_train_function,
            batched=True,
            batch_size=None,
            remove_columns=[i for i in dataset.column_names if i not in ["node_ids"]],
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=1,
        ).with_format("torch")
        clm_dataset_test = dataset.map(
            graph_dataset.preprocess_test_function,
            batched=True,
            batch_size=None,
            remove_columns=[i for i in dataset.column_names if i not in ["node_ids"]],
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=1,
        ).with_format("torch")

    accelerator.wait_for_everyone()

    train_dataset = clm_dataset_train.select(split["train"])
    val_dataset = clm_dataset_train.select(split["valid"])
    val_dataset_eval = clm_dataset_test.select(split["valid"])
    test_dataset = clm_dataset_test.select(split["test"])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=default_data_collator,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=default_data_collator,
    )
    val_loader_eval = torch.utils.data.DataLoader(
        val_dataset_eval,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=default_data_collator,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=default_data_collator,
    )

    with open(Path(f"{args.module_path}/{args.model_name}/") / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args = ModelArgs(
        w_lora=False,
        w_adapter=True,
        adapter_layer=8,
        adapter_dim=args.adapter_dim,
        adapter_len=args.adapter_len,
        lora_alpha=16,
        lora_r=8,
        num_hops=3,
        n_mp_layers=args.n_mp_layers,
        rrwp=args.rrwp,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        adapter_n_heads=args.adapter_n_heads,
        task_level=task_level[args.dataset],
        **params,
    )

    model_args.vocab_size = tokenizer.vocab_size
    # Changed: torch.cuda.BFloat16Tensor -> torch.Tensor.bfloat16
    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    base_model = Transformer(
        params=model_args,
        edge_index=edge_index,
        input_ids=original_dataset["input_ids"],
        input_attention_mask=original_dataset["attention_mask"],
    )
    torch.set_default_tensor_type(torch.FloatTensor)

    ckpt_path = Path(f"{args.module_path}/{args.model_name}/consolidated.00.pth")
    base_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)

    accelerator.print(model_args)

    param_adater, param_lora = base_model.set_trainable_params_new()

    lr_group = {"adapter": args.lr, "lora": args.lr}
    wd_group = {"adapter": args.wd, "lora": args.wd}
    accelerator.print(lr_group)
    accelerator.print(wd_group)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": param_adater,
                "lr": lr_group["adapter"],
                "weight_decay": wd_group["adapter"],
            },
            {
                "params": param_lora,
                "lr": lr_group["lora"],
                "weight_decay": wd_group["lora"],
            },
        ],
        betas=(0.9, 0.95),
    )

    trainable_params, all_params = base_model.print_trainable_params()
    accelerator.print(
        f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}"
    )
    model, train_loader, val_loader, val_loader_eval, optimizer = accelerator.prepare(
        base_model, train_loader, val_loader, val_loader_eval, optimizer
    )

    # Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss, best_val_acc = float("inf"), -float("inf")

    for epoch in range(args.num_epochs):

        model.train()
        epoch_loss, accum_loss = 0.0, 0.0

        for step, batch in enumerate(train_loader):

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(**batch)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(optimizer.param_groups[0]["params"], 0.1)
                accelerator.clip_grad_norm_(optimizer.param_groups[1]["params"], 0.1)

                if (step + 1) % args.grad_steps == 0:
                    adjust_lr(
                        optimizer.param_groups[0],
                        lr_group["adapter"],
                        step / len(train_loader) + epoch,
                        args,
                    )
                    adjust_lr(
                        optimizer.param_groups[1],
                        lr_group["lora"],
                        step / len(train_loader) + epoch,
                        args,
                    )

                optimizer.step()
                epoch_loss, accum_loss = (
                    epoch_loss + loss.item(),
                    accum_loss + loss.item(),
                )

            if (step + 1) % args.grad_steps == 0:
                adapter_lr = optimizer.param_groups[0]["lr"]
                lora_lr = optimizer.param_groups[1]["lr"]

                accelerator.log({"Adapter Lr": adapter_lr, "Lora Lr": lora_lr})
                accelerator.log({"Accum Loss": accum_loss / args.grad_steps})
                accelerator.print(f"Accum Loss: {accum_loss / args.grad_steps}")
                accum_loss = 0.0

            progress_bar.update(1)

        accelerator.print(
            f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}"
        )
        accelerator.log({"Train Loss (Epoch Mean)": epoch_loss / len(train_loader)})

        val_loss = 0.0
        samples_seen = 0
        eval_output = []
        model.eval()

        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = model(**batch)
                val_loss += loss.item()

            accelerator.print(
                f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss / len(val_loader)}"
            )
            accelerator.log({"Val Loss": val_loss / len(val_loader)})

            for step, batch in enumerate(val_loader_eval):
                kwargs = {}
                kwargs.update(
                    {
                        "node_ids": batch["node_ids"],
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"],
                        "max_new_tokens": 15,
                    }
                )

                generated_tokens = accelerator.unwrap_model(model).generate(**kwargs)
                generated_tokens_gathered = (
                    accelerator.gather(generated_tokens).cpu().numpy()
                )

                if accelerator.num_processes > 1:
                    if step == len(val_loader_eval) - 1:
                        generated_tokens_gathered = generated_tokens_gathered[
                            : len(val_loader_eval.dataset) - samples_seen
                        ]
                    else:
                        samples_seen += len(generated_tokens_gathered)
                eval_output.append(generated_tokens_gathered)

        eval_decode_output = []
        for batch_output in eval_output:
            eval_decode_output.extend(
                tokenizer.batch_decode(batch_output, skip_special_tokens=False)
            )

        eval_pred = [item.split("</s>")[0] for item in eval_decode_output]
        eval_pred = [item.split("\n\n###\n\n ")[-1] for item in eval_pred]

        eval_label = val_loader_eval.dataset["text_label"]
        pred = [_ == f"{eval_label[i]}" for i, _ in enumerate(eval_pred)]
        val_acc = sum(pred) / len(pred)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(accelerator.unwrap_model(model).cpu())
            best_epoch = epoch
            model = model.cuda()

        accelerator.print(
            f"Epoch {epoch} Val Acc {val_acc} Best Val Acc {best_val_acc} Best Epoch {best_epoch}"
        )
        accelerator.log({"val acc": val_acc})

    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    accelerator.wait_for_everyone()

    # Eval

    model, test_loader = accelerator.prepare(best_model, test_loader)

    samples_seen = 0
    eval_output = []
    model.eval()

    progress_bar_test = tqdm(range(len(test_loader)))

    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            kwargs = {}
            kwargs.update(
                {
                    "node_ids": batch["node_ids"],
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "max_new_tokens": 15,
                }
            )

            generated_tokens = accelerator.unwrap_model(model).generate(**kwargs)
            generated_tokens_gathered = (
                accelerator.gather(generated_tokens).cpu().numpy()
            )

            if accelerator.num_processes > 1:
                if step == len(test_loader) - 1:
                    generated_tokens_gathered = generated_tokens_gathered[
                        : len(test_loader.dataset) - samples_seen
                    ]
                else:
                    samples_seen += len(generated_tokens_gathered)

            eval_output.append(generated_tokens_gathered)

        progress_bar_test.update(1)

    # Post-Process
    if accelerator.is_local_main_process:
        eval_decode_output = []
        for batch_output in eval_output:
            eval_decode_output.extend(
                tokenizer.batch_decode(batch_output, skip_special_tokens=False)
            )

        eval_pred = [item.split("</s>")[0] for item in eval_decode_output]
        eval_pred = [item.split("\n\n###\n\n ")[-1] for item in eval_pred]

        eval_label = test_loader.dataset["text_label"]
        pred = [_ == f"{eval_label[i]}" for i, _ in enumerate(eval_pred)]

        acc = sum(pred) / len(pred)

        accelerator.print(f"Test Acc {acc}")
        accelerator.log({"Test Acc": acc})


if __name__ == "__main__":
    args = parse_args_llama

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
