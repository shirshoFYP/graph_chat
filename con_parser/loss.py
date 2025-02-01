import torch
import torch.nn.functional as F
from transition_system import Action
from tree import Tree


def action_loss(logits, gt_actions, label_vocab, batch_size):

    # ground truth tensors
    gt_target_nodes_list = []
    gt_parent_labels_list = []
    gt_new_labels_list = []

    for action in gt_actions:
        gt_target_nodes_list.append(action.target_node)
        gt_parent_labels_list.append(label_vocab.index(action.parent_label))
        gt_new_labels_list.append(label_vocab.index(action.new_label))

    gt_target_nodes = logits["target_node"].new_tensor(
        gt_target_nodes_list, dtype=torch.int64
    )
    gt_parent_labels = logits["parent_label"].new_tensor(
        gt_parent_labels_list, dtype=torch.int64
    )
    gt_new_labels = logits["new_label"].new_tensor(
        gt_new_labels_list, dtype=torch.int64
    )

    # calculate loss
    node_loss = F.cross_entropy(logits["target_node"], gt_target_nodes, reduction="sum")
    parent_loss = F.cross_entropy(
        logits["parent_label"], gt_parent_labels, reduction="sum"
    )
    new_label_loss = F.cross_entropy(
        logits["new_label"], gt_new_labels, reduction="sum"
    )

    loss = (node_loss + parent_loss + new_label_loss) / batch_size
    return loss


def action_seq_loss(logits, gt_actions, label_vocab, batch_size):
    subbatch_size, max_len, _ = logits["target_node"].size()
    valid_action_mask = logits["target_node"].new_zeros(
        (subbatch_size, max_len), dtype=torch.bool
    )

    for i, actions in enumerate(gt_actions):
        valid_action_mask[i, : len(actions)] = True

    node_logits = logits["target_node"][valid_action_mask]
    parent_label_logits = logits["parent_label"][valid_action_mask]
    new_label_logits = logits["new_label"][valid_action_mask]

    # ground truth tensors
    gt_target_nodes_list = []
    gt_parent_labels_list = []
    gt_new_labels_list = []

    for action_seq in gt_actions:
        for action in action_seq:
            gt_target_nodes_list.append(action.target_node)
            gt_parent_labels_list.append(label_vocab.index(action.parent_label))
            gt_new_labels_list.append(label_vocab.index(action.new_label))

    gt_target_nodes = node_logits.new_tensor(gt_target_nodes_list, dtype=torch.int64)
    gt_parent_labels = parent_label_logits.new_tensor(
        gt_parent_labels_list, dtype=torch.int64
    )
    gt_new_labels = new_label_logits.new_tensor(gt_new_labels_list, dtype=torch.int64)

    # calculate loss
    node_loss = F.cross_entropy(node_logits, gt_target_nodes, reduction="sum")
    parent_loss = F.cross_entropy(
        parent_label_logits, gt_parent_labels, reduction="sum"
    )
    new_label_loss = F.cross_entropy(new_label_logits, gt_new_labels, reduction="sum")

    loss = (node_loss + parent_loss + new_label_loss) / batch_size
    return loss
