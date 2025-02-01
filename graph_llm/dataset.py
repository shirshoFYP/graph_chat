import torch
import pickle
from transformers import LlamaTokenizer
from torch.utils.data import TensorDataset
import datasets


class GraphLLMDataset(TensorDataset):

    def __init__(self, module_path, dataset_name, tokenizer):
        self.module_path = module_path
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.preprocess_function = self.preprocess_function_original_sc_mts_sp_bgm(
            self.tokenizer, max_length=512
        )
        if self.dataset_name == "sp":
            self.dataset, self.split, self.edge_index = self.load_textual_sp()
            self.preprocess_train_function = self.preprocess_function_generator_sp(
                self.tokenizer, max_length=512
            )
            self.preprocess_test_function = self.preprocess_test_function_generator_sp(
                self.tokenizer, max_length=32
            )
        elif self.dataset_name == "mts":
            self.dataset, self.split, self.edge_index = self.load_textual_mts()
            self.preprocess_train_function = self.preprocess_function_generator_mts(
                self.tokenizer, ignore_index=-100, max_length=32
            )
            self.preprocess_test_function = self.preprocess_test_function_generator_mts(
                self.tokenizer, max_length=32
            )
        elif self.dataset_name == "sc":
            self.dataset, self.split, self.edge_index = self.load_textual_sc()
            self.preprocess_train_function = self.preprocess_function_generator_sc(
                self.tokenizer, ignore_index=-100, max_length=32
            )
            self.preprocess_test_function = self.preprocess_test_function_generator_sc(
                self.tokenizer, max_length=32
            )
        elif self.dataset_name == "bgm":
            self.dataset, self.split, self.edge_index = self.load_textual_bgm()
            self.preprocess_train_function = self.preprocess_function_generator_bgm(
                self.tokenizer, ignore_index=-100, max_length=32
            )
            self.preprocess_test_function = self.preprocess_test_function_generator_bgm(
                self.tokenizer, max_length=32
            )
        else:
            raise ValueError(f"Dataset {self.dataset_name} not found.")

    def get_dataset(self):
        return self.dataset, self.split, self.edge_index

    def load_textual_sp(self):
        dataset = datasets.load_from_disk(f"{self.module_path}/dataset/sp")
        split = pickle.load(open(f"{self.module_path}/dataset/sp/sp_split.pkl", "rb"))
        edge_index = pickle.load(
            open(f"{self.module_path}/dataset/sp/sp_edge.pkl", "rb")
        )
        return dataset, split, edge_index

    def load_textual_mts(self):
        dataset = datasets.load_from_disk(f"{self.module_path}/dataset/mts")
        split = pickle.load(open(f"{self.module_path}/dataset/mts/mts_split.pkl", "rb"))
        edge_index = pickle.load(
            open(f"{self.module_path}/dataset/mts/mts_edge.pkl", "rb")
        )
        return dataset, split, edge_index

    def load_textual_sc(self):
        dataset = datasets.load_from_disk(f"{self.module_path}/dataset/sc")
        split = pickle.load(open(f"{self.module_path}/dataset/sc/sc_split.pkl", "rb"))
        edge_index = pickle.load(
            open(f"{self.module_path}/dataset/sc/sc_edge.pkl", "rb")
        )
        return dataset, split, edge_index

    def load_textual_bgm(self):
        dataset = datasets.load_from_disk(f"{self.module_path}/dataset/bgm")
        split = pickle.load(open(f"{self.module_path}/dataset/bgm/bgm_split.pkl", "rb"))
        edge_index = pickle.load(
            open(f"{self.module_path}/dataset/bgm/bgm_edge.pkl", "rb")
        )
        return dataset, split, edge_index

    def preprocess_function_original_sc_mts_sp_bgm(self, tokenizer, max_length=512):
        def preprocess_function(examples):
            desc = [f"{d}" for d in examples["desc"]]
            desc_ids = tokenizer(
                desc,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return desc_ids

        return preprocess_function

    def preprocess_function_generator_mts(
        self, tokenizer, ignore_index=-100, max_length=32
    ):
        def preprocess_function(examples):
            prompts = [
                f"What is the maximum sum of age of a triplet composed of person {name}, "
                f"their friends and friends of friends?"
                for name in examples["name"]
            ]
            completion = [f"The maximum sum is {l}." for l in examples["label"]]

            instruction = f"\n\n###\n\n"
            instruction = tokenizer.encode(instruction, add_special_tokens=False)

            model_inputs = tokenizer(prompts, add_special_tokens=False)
            labels = tokenizer(completion, add_special_tokens=False)

            batch_size = len(examples["node_ids"])

            for i in range(batch_size):
                # Add bos & eos token
                sample_input_ids = [tokenizer.bos_token_id] + model_inputs["input_ids"][
                    i
                ]
                label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]

                p_max_length = max_length - len(label_input_ids) - len(instruction)
                sample_input_ids = sample_input_ids[:p_max_length] + instruction

                model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
                labels["input_ids"][i] = [ignore_index] * len(
                    sample_input_ids
                ) + label_input_ids
                model_inputs["attention_mask"][i] = [1] * len(
                    model_inputs["input_ids"][i]
                )

            for i in range(batch_size):
                sample_input_ids = model_inputs["input_ids"][i]
                label_input_ids = labels["input_ids"][i]
                model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
                ) + sample_input_ids
                model_inputs["attention_mask"][i] = [0] * (
                    max_length - len(sample_input_ids)
                ) + model_inputs["attention_mask"][i]
                labels["input_ids"][i] = [ignore_index] * (
                    max_length - len(sample_input_ids)
                ) + label_input_ids
                model_inputs["input_ids"][i] = torch.tensor(
                    model_inputs["input_ids"][i]
                )
                model_inputs["attention_mask"][i] = torch.tensor(
                    model_inputs["attention_mask"][i]
                )
                labels["input_ids"][i] = torch.tensor(labels["input_ids"][i])

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return preprocess_function

    def preprocess_function_generator_sc(
        self, tokenizer, ignore_index=-100, max_length=32
    ):
        def preprocess_function(examples):
            prompts = [
                f"How many carbon-carbon-oxygen triangles containing Atom #1 are in the "
                f"molecule?"
                for _ in examples["desc"]
            ]
            completion = [f"{l} C-C-O triangles." for l in examples["label"]]

            instruction = f"\n\n###\n\n"
            instruction = tokenizer.encode(instruction, add_special_tokens=False)

            model_inputs = tokenizer(prompts, add_special_tokens=False)
            labels = tokenizer(completion, add_special_tokens=False)

            batch_size = len(examples["node_ids"])

            for i in range(batch_size):
                # Add bos & eos token
                sample_input_ids = [tokenizer.bos_token_id] + model_inputs["input_ids"][
                    i
                ]
                label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]

                p_max_length = max_length - len(label_input_ids) - len(instruction)
                sample_input_ids = sample_input_ids[:p_max_length] + instruction

                model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
                labels["input_ids"][i] = [ignore_index] * len(
                    sample_input_ids
                ) + label_input_ids
                model_inputs["attention_mask"][i] = [1] * len(
                    model_inputs["input_ids"][i]
                )

            for i in range(batch_size):
                sample_input_ids = model_inputs["input_ids"][i]
                label_input_ids = labels["input_ids"][i]
                model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
                ) + sample_input_ids
                model_inputs["attention_mask"][i] = [0] * (
                    max_length - len(sample_input_ids)
                ) + model_inputs["attention_mask"][i]
                labels["input_ids"][i] = [ignore_index] * (
                    max_length - len(sample_input_ids)
                ) + label_input_ids
                model_inputs["input_ids"][i] = torch.tensor(
                    model_inputs["input_ids"][i]
                )
                model_inputs["attention_mask"][i] = torch.tensor(
                    model_inputs["attention_mask"][i]
                )
                labels["input_ids"][i] = torch.tensor(labels["input_ids"][i])

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return preprocess_function

    def preprocess_function_generator_sp(
        self, tokenizer, ignore_index=-100, max_length=32
    ):
        def preprocess_function(examples):
            prompts = [
                f"Starting from wormhole #1, how much dark matter do you need at "
                f"minimum to reach wormhole #2?"
                for dest in examples["dest"]
            ]

            completion = [f"{l}" for l in examples["label"]]
            instruction = f"\n\n###\n\n"
            instruction = tokenizer.encode(instruction, add_special_tokens=False)

            model_inputs = tokenizer(prompts, add_special_tokens=False)
            labels = tokenizer(completion, add_special_tokens=False)

            batch_size = len(examples["node_ids"])

            for i in range(batch_size):
                # Add bos & eos token
                sample_input_ids = [tokenizer.bos_token_id] + model_inputs["input_ids"][
                    i
                ]
                label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]

                p_max_length = max_length - len(label_input_ids) - len(instruction)
                sample_input_ids = sample_input_ids[:p_max_length] + instruction

                model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
                labels["input_ids"][i] = [ignore_index] * len(
                    sample_input_ids
                ) + label_input_ids
                model_inputs["attention_mask"][i] = [1] * len(
                    model_inputs["input_ids"][i]
                )

            for i in range(batch_size):
                sample_input_ids = model_inputs["input_ids"][i]
                label_input_ids = labels["input_ids"][i]
                model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
                ) + sample_input_ids
                model_inputs["attention_mask"][i] = [0] * (
                    max_length - len(sample_input_ids)
                ) + model_inputs["attention_mask"][i]
                labels["input_ids"][i] = [ignore_index] * (
                    max_length - len(sample_input_ids)
                ) + label_input_ids
                model_inputs["input_ids"][i] = torch.tensor(
                    model_inputs["input_ids"][i]
                )
                model_inputs["attention_mask"][i] = torch.tensor(
                    model_inputs["attention_mask"][i]
                )
                labels["input_ids"][i] = torch.tensor(labels["input_ids"][i])

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return preprocess_function

    def preprocess_function_generator_bgm(
        self, tokenizer, ignore_index=-100, max_length=32
    ):
        def preprocess_function(examples):
            prompts = [
                f"For most how many applicants can find the job they are interested in?"
                for _ in examples["desc"]
            ]
            completion = [f"{l} applicants." for l in examples["label"]]

            instruction = f"\n\n###\n\n"
            instruction = tokenizer.encode(instruction, add_special_tokens=False)

            model_inputs = tokenizer(prompts, add_special_tokens=False)
            labels = tokenizer(completion, add_special_tokens=False)

            batch_size = len(examples["node_ids"])

            for i in range(batch_size):
                # Add bos & eos token
                sample_input_ids = [tokenizer.bos_token_id] + model_inputs["input_ids"][
                    i
                ]
                label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]

                p_max_length = max_length - len(label_input_ids) - len(instruction)
                sample_input_ids = sample_input_ids[:p_max_length] + instruction

                model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
                labels["input_ids"][i] = [ignore_index] * len(
                    sample_input_ids
                ) + label_input_ids
                model_inputs["attention_mask"][i] = [1] * len(
                    model_inputs["input_ids"][i]
                )

            for i in range(batch_size):
                sample_input_ids = model_inputs["input_ids"][i]
                label_input_ids = labels["input_ids"][i]
                model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
                ) + sample_input_ids
                model_inputs["attention_mask"][i] = [0] * (
                    max_length - len(sample_input_ids)
                ) + model_inputs["attention_mask"][i]
                labels["input_ids"][i] = [ignore_index] * (
                    max_length - len(sample_input_ids)
                ) + label_input_ids
                model_inputs["input_ids"][i] = torch.tensor(
                    model_inputs["input_ids"][i]
                )
                model_inputs["attention_mask"][i] = torch.tensor(
                    model_inputs["attention_mask"][i]
                )
                labels["input_ids"][i] = torch.tensor(labels["input_ids"][i])

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return preprocess_function

    def preprocess_test_function_generator_sp(self, tokenizer, max_length=32):
        def preprocess_function(examples):
            prompts = [
                f"Starting from wormhole #1, how much dark matter do you need at "
                f"minimum to reach wormhole #2?"
                for dest in examples["dest"]
            ]
            model_inputs = tokenizer(prompts, add_special_tokens=False)

            instruction = f"\n\n###\n\n"
            instruction = tokenizer.encode(instruction, add_special_tokens=False)

            batch_size = len(examples["node_ids"])

            for i in range(batch_size):
                sample_input_ids = (
                    [tokenizer.bos_token_id]
                    + model_inputs["input_ids"][i][: max_length - len(instruction) - 1]
                    + instruction
                )

                model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
                ) + sample_input_ids

                model_inputs["attention_mask"][i] = [0] * (
                    max_length - len(sample_input_ids)
                ) + [1] * len(sample_input_ids)
                model_inputs["input_ids"][i] = torch.tensor(
                    model_inputs["input_ids"][i]
                )
                model_inputs["attention_mask"][i] = torch.tensor(
                    model_inputs["attention_mask"][i]
                )

            model_inputs["text_label"] = [f"{l}" for l in examples["label"]]
            return model_inputs

        return preprocess_function

    def preprocess_test_function_generator_mts(self, tokenizer, max_length=32):
        def preprocess_function(examples):
            prompts = [
                f"What is the maximum sum of age of a triplet composed of person {name}, "
                f"their friends and friends of friends?"
                for name in examples["name"]
            ]

            model_inputs = tokenizer(prompts, add_special_tokens=False)

            instruction = f"\n\n###\n\n"
            instruction = tokenizer.encode(instruction, add_special_tokens=False)

            batch_size = len(examples["node_ids"])

            for i in range(batch_size):
                sample_input_ids = (
                    [tokenizer.bos_token_id]
                    + model_inputs["input_ids"][i][: max_length - len(instruction) - 1]
                    + instruction
                )

                model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
                ) + sample_input_ids

                model_inputs["attention_mask"][i] = [0] * (
                    max_length - len(sample_input_ids)
                ) + [1] * len(sample_input_ids)
                model_inputs["input_ids"][i] = torch.tensor(
                    model_inputs["input_ids"][i]
                )
                model_inputs["attention_mask"][i] = torch.tensor(
                    model_inputs["attention_mask"][i]
                )

            model_inputs["text_label"] = [
                f"The maximum sum is {l}." for l in examples["label"]
            ]
            return model_inputs

        return preprocess_function

    def preprocess_test_function_generator_bgm(self, tokenizer, max_length=32):
        def preprocess_function(examples):
            prompts = [
                f"For most how many applicants can find the job they are interested in?"
                for _ in examples["desc"]
            ]

            model_inputs = tokenizer(prompts, add_special_tokens=False)

            instruction = f"\n\n###\n\n"
            instruction = tokenizer.encode(instruction, add_special_tokens=False)

            batch_size = len(examples["node_ids"])

            for i in range(batch_size):
                sample_input_ids = (
                    [tokenizer.bos_token_id]
                    + model_inputs["input_ids"][i][: max_length - len(instruction) - 1]
                    + instruction
                )

                model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
                ) + sample_input_ids

                model_inputs["attention_mask"][i] = [0] * (
                    max_length - len(sample_input_ids)
                ) + [1] * len(sample_input_ids)
                model_inputs["input_ids"][i] = torch.tensor(
                    model_inputs["input_ids"][i]
                )
                model_inputs["attention_mask"][i] = torch.tensor(
                    model_inputs["attention_mask"][i]
                )

            model_inputs["text_label"] = [f"{l} applicants." for l in examples["label"]]
            return model_inputs

        return preprocess_function

    def preprocess_test_function_generator_sc(self, tokenizer, max_length=32):
        def preprocess_function(examples):
            prompts = [
                f"How many carbon-carbon-oxygen triangles containing Atom #1 are in the "
                f"molecule?"
                for _ in examples["desc"]
            ]

            model_inputs = tokenizer(prompts, add_special_tokens=False)

            instruction = f"\n\n###\n\n"
            instruction = tokenizer.encode(instruction, add_special_tokens=False)

            batch_size = len(examples["node_ids"])

            for i in range(batch_size):
                sample_input_ids = (
                    [tokenizer.bos_token_id]
                    + model_inputs["input_ids"][i][: max_length - len(instruction) - 1]
                    + instruction
                )

                model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
                ) + sample_input_ids

                model_inputs["attention_mask"][i] = [0] * (
                    max_length - len(sample_input_ids)
                ) + [1] * len(sample_input_ids)
                model_inputs["input_ids"][i] = torch.tensor(
                    model_inputs["input_ids"][i]
                )
                model_inputs["attention_mask"][i] = torch.tensor(
                    model_inputs["attention_mask"][i]
                )

            model_inputs["text_label"] = [
                f"{l} C-C-O triangles." for l in examples["label"]
            ]
            return model_inputs

        return preprocess_function
