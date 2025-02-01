from copy import deepcopy
from typing import NamedTuple
from tree import Tree, Label, InternalParseNode, LeafParseNode, DUMMY_LABEL


class Action(NamedTuple):
    action_type: str
    target_node: int
    parent_label: Label
    new_label: Label

    def normalize(self):
        return Action(
            self.action_type,
            self.target_node,
            tuple(self.parent_label),
            tuple(self.new_label),
        )


class AttachJuxtapose:
    """The attack-juxtapose transition system for constituency parsing"""

    @staticmethod
    def oracle_actions(tree, immutable):
        """Appendix A.1"""
        if tree is None:
            return []
        if immutable:
            tree = deepcopy(tree)
        last_action, last_tree = AttachJuxtapose._undo_last_action(tree)
        return AttachJuxtapose.oracle_actions(last_tree, False) + [last_action]

    @staticmethod
    def execute(tree, action, pos, tag, word, immutable):
        """Section 3"""
        new_leaf = LeafParseNode(pos, tag, word)
        if immutable:
            tree = deepcopy(tree)
        action_type, target_node_idx, parent_label, new_label = action
        new_subtree = (
            new_leaf
            if parent_label == DUMMY_LABEL
            else InternalParseNode(parent_label, [new_leaf], None)
        )

        target_node = tree
        if target_node is not None:
            for _ in range(target_node_idx):
                next_node = target_node.children[-1]
                if not isinstance(next_node, InternalParseNode):  # truncate
                    break
                target_node = next_node

        if action_type == "attach":
            return AttachJuxtapose._execute_attach(tree, target_node, new_subtree)
        assert (
            action_type == "juxtapose"
            and target_node is not None
            and new_label != DUMMY_LABEL
        )
        return AttachJuxtapose._execute_juxtapose(
            tree, target_node, new_subtree, new_label
        )

    @staticmethod
    def actions_to_tree(word_seq, tag_seq, action_seq):
        tree = None
        for pos, (word, tag, action) in enumerate(zip(word_seq, tag_seq, action_seq)):
            tree = AttachJuxtapose.execute(tree, action, pos, tag, word, False)
        assert tree is not None
        return tree

    @staticmethod
    def _undo_last_action(tree):
        last_leaf = tree.last_leaf()
        siblings = last_leaf.get_siblings()

        last_subtree = []
        if len(siblings) > 0:
            last_subtree = last_leaf
            last_subtree_siblings = siblings
            parent_label = DUMMY_LABEL
        else:
            assert last_leaf.parent is not None
            last_subtree_siblings = (
                last_subtree.get_siblings() if not last_subtree.is_root() else []
            )
            parent_label = last_subtree.label
        if last_subtree.is_root():
            return Action("attach", 0, parent_label, ()), None

        if len(last_subtree_siblings) == 1 and isinstance(
            last_subtree_siblings[0], InternalParseNode
        ):
            assert last_subtree.parent is not None
            new_label = last_subtree.parent.label
            target_node = last_subtree_siblings[0]
            grand_parent = last_subtree.parent.parent
            if grand_parent is None:
                tree = target_node
                target_node.parent = None
            else:
                grand_parent.children = [
                    target_node if child is last_subtree.parent else child
                    for child in grand_parent.children
                ]
                target_node.parent = grand_parent
            return Action("juxtapose", target_node.depth, parent_label, new_label), tree
        target_node = last_subtree.parent
        target_node.children.remove(last_subtree)
        return Action("attach", target_node.depth, parent_label, ()), tree

    @staticmethod
    def _execute_attach(tree, target_node, new_subtree):
        if target_node is None:
            assert (
                isinstance(new_subtree, InternalParseNode)
                and new_subtree.left == 0
                and new_subtree.right == 1
            )
            tree = new_subtree
        else:
            target_node.child.append(new_subtree)
            new_subtree.parent = target_node
            AttachJuxtapose._update_right(target_node)
        return tree

    @staticmethod
    def _execute_juxtapose(tree, target_node, new_subtree, new_label):
        parent = target_node.parent
        new_node = InternalParseNode(new_label, children=[target_node, new_subtree])
        if parent is None:
            assert tree is target_node
            tree = new_node
        else:
            assert target_node is parent.children[-1]
            parent.children = parent.children[:-1] + [new_node]
            new_node.parent = parent
            AttachJuxtapose._update_right(parent)
        return tree

    @staticmethod
    def _update_right(node):
        cur = node
        while cur is not None:
            cur.right = cur.children[-1].right
            cur = cur.parent
