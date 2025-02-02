"""
Parse Trees
"""

from lark import Lark, Transformer
from lark.lexer import Token
from typing import Tuple, Optional


class ParseNode:

    def __init__(self, left: int, right: int, parent=None):
        self.left = left
        self.right = right
        self.parent = parent

    def is_root(self):
        return self.parent is None

    def linearize(self):
        raise NotImplementedError

    def iter_leaves(self):
        raise NotImplementedError

    def traverse_preorder(self, callback=None):
        raise NotImplementedError

    def traverse_post(self, callback=None):
        raise NotImplementedError

    def get_siblings(self):
        assert self.parent is not None
        return [sibling for sibling in self.parent.children if sibling is not self]

    @property
    def depth(self):
        d = -1
        cur = self
        while cur is not None:
            d += 1
            cur = cur.parent
        return d


Label = Tuple[str, ...]

DUMMY_LABEL = ()


class InternalParseNode(ParseNode):

    def __init__(self, label, children, parent=None):
        super().__init__(children[0].left, children[-1].right, parent)
        self.label = label
        self.children = children
        for child in children:
            child.parent = self

    def linearize(self):
        return (
            " ".join(["(" + sublabel for sublabel in self.label])
            + " "
            + " ".join(child.linearize() for child in self.children)
            + ")" * len(self.label)
        )

    def traverse_preorder(self, callback) -> None:
        callback(self)
        for c in self.children:
            c.traverse_preorder(callback)

    def traverse_post(self, callback) -> None:
        for c in self.children:
            c.traverse_post(callback)
        callback(self)

    def last_leaf(self):
        cur = self
        while isinstance(cur, InternalParseNode):
            cur = cur.children[-1]
        assert isinstance(cur, LeafParseNode)
        return cur

    def iter_rightmost_chain(self):
        cur = self
        while isinstance(cur, InternalParseNode):
            yield cur
            if isinstance(cur.children[-1], InternalParseNode):
                cur = cur.children[-1]
            else:
                break

    def node_on_rightmost_chain(self, i):
        for j, node in enumerate(self.iter_rightmost_chain()):
            if j == i:
                return node

        import pdb

        pdb.set_trace()
        raise IndexError

    def is_well_formed(self):
        good = True

        def check(node):
            nonlocal good
            good &= node.is_root() or isinstance(node.parent, InternalParseNode)
            if isinstance(node, LeafParseNode):
                good &= (
                    isinstance(node.left, int)
                    and isinstance(node.right, int)
                    and 0 <= node.left
                    and node.right == node.left + 1
                )
                good &= isinstance(node.tag, str) and isinstance(node.word, str)
            else:
                assert isinstance(node, InternalParseNode)
                good &= (
                    isinstance(node.label, tuple)
                    and node.label is not None
                    and all(isinstance(sublabel, str) for sublabel in node.label)
                )
                good &= isinstance(node.children, list)
                good &= len(node.children) > 1 or isinstance(
                    node.children[0], LeafParseNode
                )
                for child in node.children:
                    good &= isinstance(child, ParseNode) and child.parent is node
                good &= all(
                    left.right == right.left
                    for left, right in zip(node.children, node.children[1:])
                )

        self.traverse_preorder(check)
        return good

    def iter_leaves(self):
        for child in self.children:
            yield from child.iter_leaves()


Tree = Optional[InternalParseNode]


def rightmost_chain_length(tree: Tree) -> int:
    if tree is None:
        return 0
    return sum(1 for _ in tree.iter_rightmost_chain())


class LeafParseNode(ParseNode):

    def __init__(self, pos: int, tag: str, word: str, parent=None):
        super().__init__(pos, pos + 1, parent)
        self.pos = pos
        self.tag = tag
        self.word = word

    def linearize(self):
        return f"({self.tag} {self.word})"

    def iter_leaves(self):
        yield self

    def traverse_preorder(self, callback=None) -> None:
        callback(self)

    def traverse_post(self, callback=None) -> None:
        callback(self)


class TreeBuilder(Transformer):
    "Construct parse trees from S-expressions used in the datasets"
    pos: int = 0

    def sexp(self, children):
        children = [c for c in children if c is not None]
        if len(children) < 2:
            return None
        if len(children) == 2 and isinstance(children[1], Token):
            if children[0].value == "-NONE-":
                return None
            leaf = LeafParseNode(self.pos, children[0].value, children[1].value, None)
            self.pos += 1
            return leaf
        elif children[0].value == "TOP":
            # remove TOP
            self.pos = 0
            return children[1]
        else:
            sublabel = children[0].value
            if len(children) == 2 and isinstance(children[1], InternalParseNode):
                # unary
                parent = children[1]
                parent.label = (sublabel,) + parent.label
            else:
                parent = InternalParseNode((sublabel,), children[1:], None)
            return parent


class TreeBatch:
    def __init__(self, trees):
        self.trees = trees

    @staticmethod
    def from_file(filename):
        sexp_ebnf = """
            sexp : "(" SYMBOL (sexp|SYMBOL)* ")"
            SYMBOL : /[^\(\)\s]+/
            %import common.WS
            %ignore WS
        """
        sexp_parser = Lark(sexp_ebnf, start="sexp", parser="lalr")
        tree_builder = TreeBuilder()
        trees = [
            tree_builder.transform(sexp_parser.parse(tree_sexp.strip()))
            for tree_sexp in open(filename)
        ]
        return TreeBatch(trees)

    @staticmethod
    def empty_trees(n):
        return TreeBatch([None] * n)

    def __getitem__(self, i):
        return self.trees[i]

    def __len__(self):
        return len(self.trees)

    def traverse_preorder(self, callback=None):
        for tree in self.trees:
            if tree is not None:
                tree.traverse_preorder(callback)


if __name__ == "__main__":
    import sys
    from pprint import pprint

    trees = TreeBatch.from_file(sys.argv[1])
    for tree in trees:
        pprint(tree.linearize())
        print(tree.is_well_formed())
        print()
