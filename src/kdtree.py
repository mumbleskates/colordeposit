# coding=utf-8


def uniform_split(envelope, dim):
    """
    Determines which direction an item splits when traversing down.

    Returns midpoint for the specified dimension
    """
    lower, upper = envelope[dim]
    return (lower + upper) / 2


class ImplicitKdTree(object):
    """Mutable k-d tree implementation optimized for uniform distributions.
    Pathological distributions will unavoidably unbalance the tree."""
    __slots__ = ('items', 'head', 'k', 'envelope', 'splitter')

    def __init__(self, k, envelope, splitter=uniform_split):
        """
        :param k: dimensionality
        :param envelope: ((min, max), ...) for each dimension
        """
        self.items = {}
        self.head = None
        self.k = k
        self.envelope = envelope
        self.splitter = splitter

    def __len__(self):
        return len(self.items)

    def get(self, key):
        return self.items[key].val

    def set(self, key, value):
        if key in self.items:
            self.remove(key)
        if self.head:
            envelope = list(self.envelope)
            current = self.head
            depth = 0
            while True:
                dim = depth % self.k
                mid = self.splitter(envelope, dim)
                if value[dim] < mid:
                    envelope[dim] = (envelope[dim][0], mid)
                    depth += 1
                    # traverse left
                    if current.left:
                        # traverse down
                        current = current.left
                        continue
                    else:
                        self.items[key] = current.left = KdNode(key, value, current, depth)
                        break
                else:
                    envelope[dim] = (mid, envelope[dim][1])
                    depth += 1
                    # traverse right
                    if current.right:
                        # traverse down
                        current = current.right
                        continue
                    else:
                        self.items[key] = current.right = KdNode(key, value, current, depth)
                        break

            # update tree depth in parents
            while current:
                if current.depth >= depth:
                    return
                current.depth = depth
                current = current.parent

        else:
            self.items[key] = self.head = KdNode(key, value, None, 0)

    def remove(self, key):
        current = self.items.pop(key)
        if not self.items:  # this was the last item
            self.head = None
            return
        replacement, popped_key, popped_val = current.pop_deepest()
        if replacement is None:
            # we need to delete ourselves from the parent
            parent = current.parent
            if parent.left is current:  # we are on the left side
                parent.left = None
            else:  # we are on the right side
                parent.right = None
        else:
            # replacement case
            current.key, current.val = popped_key, popped_val
            self.items[popped_key] = current

    def nearest(self, value, max_sq_dist=float('inf')):
        """
        Return the nearest item to the given value.

        Returns (key, value, squared_distance) tuple. If the tree is empty, returns (None, None, infinity)
        """
        if not self:
            return None, None, float('inf')

        best_sqd = max_sq_dist
        best_node = None

        def search(node, envelope, depth):
            nonlocal best_sqd, best_node
            sqd = sum((a - b)**2 for a, b in zip(value, node.val))
            # update best
            if sqd < best_sqd:
                best_sqd, best_node = sqd, node
            # prepare to traverse down
            dim = depth % self.k
            low_up = lower, upper = envelope[dim]
            mid = self.splitter(envelope, dim)
            depth += 1
            # traverse near side first
            if value[dim] < mid:
                if node.right:
                    envelope[dim] = (lower, mid)
                    search(node.right, envelope, depth)
                # traverse other side if needed
                if node.left and (mid - value[dim])**2 < best_sqd:
                    envelope[dim] = (mid, upper)
                    search(node.left, envelope, depth)
            else:
                if node.left:
                    envelope[dim] = (mid, upper)
                    search(node.left, envelope, depth)
                # traverse other side if needed
                if node.right and (mid - value[dim])**2 < best_sqd:
                    envelope[dim] = (lower, mid)
                    search(node.right, envelope, depth)
            # revert mutable envelope
            envelope[dim] = low_up

        search(self.head, list(self.envelope), 0)
        if best_node:
            return best_node.key, best_node.val, best_sqd
        else:
            return None, None, float('inf')


class KdNode(object):
    __slots__ = ('key', 'val', 'parent', 'depth', 'left', 'right')

    def __init__(self, key, val, parent, depth):
        self.key = key
        self.val = val
        self.parent = parent
        self.depth = depth  # depth of deepest node in this subtree
        self.left = None
        self.right = None

    def pop_deepest(self):
        """
        Pop off the deepest child node in this tree.

        Returns (replacement, key, value) tuple:
            replacement: either this node or None if this node was popped
            key, value: key/value of the node that was popped
        """
        if self.left is None:
            if self.right is None:
                # this is a leaf node
                # before returning, adjust tree depths downwards
                self.depth -= 1  # this node will be deleted
                current, parent = self, self.parent
                if parent and parent.left and parent.right:
                    pass  # if parent has other children, nothing happens
                else:
                    while parent and parent.depth > self.depth:
                        # check against the other side
                        if parent.left is current:
                            # we can reduce parent's depth but not below the depth of the other side
                            if parent.right is None or parent.right.depth <= self.depth:
                                parent.depth = self.depth
                            else:  # we can't decrease depth, so we are done
                                break
                        else:  # same on other side
                            if parent.left is None or parent.left.depth <= self.depth:
                                parent.depth = self.depth
                            else:
                                break
                        # traverse upwards
                        current, parent = parent, parent.parent
                return None, self.key, self.val
        else:
            if self.right is None or self.right.depth < self.left.depth:
                # left is deeper/only
                self.left, key, val = self.left.pop_deepest()
                return self, key, val
        # right is deeper/only
        self.right, key, val = self.right.pop_deepest()
        return self, key, val
