# coding=utf-8
import random

import pytest

from kdtree import ImplicitKdTree


@pytest.mark.parametrize('k', range(1, 8))
def test_fuzz(k, verbose=False, size=5000):
    tree = ImplicitKdTree(k, tuple((0, 1) for _ in range(k)))
    keys = list(range(size))
    each = size // 15

    full_count = 0
    for step in ('adding nodes', 'moving nodes'):
        for i, key in enumerate(keys):
            assert len(tree) == full_count or i
            tree.set(key, tuple(random.random() for _ in range(k)))
            if key % each == 0 or (full_count or i) < 10:
                verify(tree)
        assert len(tree) == len(keys)
        verify(tree)
        full_count = len(keys)
        if verbose:
            print(f'Max depth for {k}-d tree with {len(keys)} nodes after {step}: {tree.head.depth}')
            print(f'Tree unbalance: {average_depth(tree) / (perfect_average_depth(len(tree)) or 1)}')
        random.shuffle(keys)

    for i, key in enumerate(keys):
        assert len(tree) == len(keys) - i
        tree.remove(key)
        if i % each == 0 or len(keys) - i < 10:
            verify(tree)
    assert len(tree) == 0
    verify(tree)


def verify(tree):
    nodecount = 0

    def check(node, envelope, depth):
        nonlocal nodecount
        nodecount += 1

        dim = depth % tree.k

        # check items
        assert tree.items[node.key] is node

        # check envelope
        for dval, (lower, upper) in zip(node.val, envelope):
            assert lower <= dval <= upper

        # check depth
        if node.left is None and node.right is None:
            assert node.depth == depth
        else:
            assert node.depth == max(0 if n is None else n.depth for n in (node.left, node.right))

            # recurse downwards
            low_up = lower, upper = envelope[dim]
            mid = tree.splitter(envelope, dim)
            assert node.mid == mid
            if node.left:
                envelope[dim] = (lower, mid)
                check(node.left, envelope, depth + 1)
            if node.right:
                envelope[dim] = (mid, upper)
                check(node.right, envelope, depth + 1)
            # revert mutable envelope
            envelope[dim] = low_up

    if tree.head is not None:
        check(tree.head, list(tree.envelope), 0)

    assert len(tree) == nodecount


def average_depth(tree):
    num = 0
    denom = 0

    def count(node, depth):
        nonlocal num, denom
        num += depth
        denom += 1
        if node.left:
            count(node.left, depth + 1)
        if node.right:
            count(node.right, depth + 1)

    if tree:
        count(tree.head, 0)
    return num / (denom or 1)


def perfect_average_depth(node_count, branch_factor=2):
    if node_count == 0:
        return 0
    remaining_nodes = node_count
    num = 0
    level = 0
    while remaining_nodes:
        level_size = min(remaining_nodes, branch_factor**level)
        num += level * level_size
        remaining_nodes -= level_size
        level += 1
    return num / node_count


if __name__ == '__main__':
    verify = lambda tree: None
    for k in range(1, 8):
        test_fuzz(k, verbose=True, size=50000)
