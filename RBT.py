# Red Black Tree implementation adapted from
# https://algorithmtutor.com/Data-Structures/Tree/Red-Black-Trees/

import sys
from termcolor import colored
import numpy as np

# WARNING: for buckets, data is not only (\tilde{c}^B)
# but (\tilde{c}^B, B) for identification

# TODO:
# - modify insert and delete to take info into account
# - add deletebelow and deleteabove functions
# (might be hard to fix tree after this)

# data structure that represents a node in the tree


class Node():
    def __init__(self, data, info):
        self.data = data  # holds the key
        self.parent = None  # pointer to the parent
        self.info = info
        self.right_info = np.zeros_like(info)
        self.left_info = np.zeros_like(info)
        self.blackheight = 0
        self.left = None  # pointer to left child
        self.right = None  # pointer to right child
        self.color = 1  # 1 . Red, 0 . Black


# Null leaf
TNULL = Node(0, np.array([[0.],[0.]]))
TNULL.color = 0
TNULL.left = None
TNULL.right = None
TNULL.blackheight = 0

# class RedBlackTree implements the operations in Red Black Tree


class RedBlackTree():
    def __init__(self):
        self.root = TNULL

    def __pre_order_helper(self, node):
        if node != TNULL:
            sys.stdout.write(node.data + " ")
            self.__pre_order_helper(node.left)
            self.__pre_order_helper(node.right)

    def __in_order_helper(self, node):
        if node != TNULL:
            self.__in_order_helper(node.left)
            sys.stdout.write(node.data + " ")
            self.__in_order_helper(node.right)

    def __post_order_helper(self, node):
        if node != TNULL:
            self.__post_order_helper(node.left)
            self.__post_order_helper(node.right)
            sys.stdout.write(node.data + " ")

    def __search_tree_helper(self, node, key):
        if node == TNULL or key == node.data:
            return node

        if key < node.data:
            return self.__search_tree_helper(node.left, key)
        return self.__search_tree_helper(node.right, key)

    # fix the rb tree modified by the delete operation
    def __fix_delete(self, x):
        while x != self.root and x.color == 0:
            if x == x.parent.left:
                s = x.parent.right
                if s.color == 1:
                    # case 3.1
                    s.color = 0
                    x.parent.color = 1
                    self.left_rotate(x.parent)
                    x.parent.blackheight -= 1
                    s = x.parent.right

                if s.left.color == 0 and s.right.color == 0:
                    # case 3.2
                    s.color = 1
                    s.blackheight -= 1
                    x.parent.blackheight -= 1
                    x = x.parent
                else:
                    if s.right.color == 0:
                        # case 3.3
                        s.left.color = 0
                        s.color = 1
                        self.right_rotate(s)
                        s.blackheight -= 1
                        s.parent.blackheight += 1
                        s = x.parent.right

                    # case 3.4
                    s.color = x.parent.color
                    if s.color == 0:
                        s.blackheight += 1
                    if x.parent.color == 1:
                        x.parent.color = 0
                    else:
                        x.parent.blackheight -= 1
                    s.right.color = 0
                    s.right.blackheight += 1
                    self.left_rotate(x.parent)
                    x = self.root
            else:
                s = x.parent.left
                if s.color == 1:
                    # case 3.1
                    s.color = 0
                    x.parent.color = 1
                    self.right_rotate(x.parent)
                    x.parent.blackheight -= 1
                    s = x.parent.left

                if s.left.color == 0 and s.right.color == 0:
                    # case 3.2
                    s.color = 1
                    s.blackheight -= 1
                    x.parent.blackheight -= 1
                    x = x.parent
                else:
                    if s.left.color == 0:
                        # case 3.3
                        s.right.color = 0
                        s.color = 1
                        self.left_rotate(s)
                        s.blackheight -= 1
                        s.parent.blackheight += 1
                        s = x.parent.left

                    # case 3.4
                    s.color = x.parent.color
                    if s.color == 0:
                        s.blackheight += 1
                    if x.parent.color == 1:
                        x.parent.color = 0
                    else:
                        x.parent.blackheight -= 1
                    s.left.color = 0
                    s.left.blackheight += 1
                    self.right_rotate(x.parent)
                    x = self.root
        if x.color == 1:
            x.color = 0
            x.blackheight += 1

    def __rb_transplant(self, u, v):
        if u.parent == None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def __delete_node_helper(self, node, key, info):
        # find the node containing key
        z = TNULL
        while node != TNULL:
            if node.data == key:
                z = node
                break

            if node.data < key:
                node.right_info -= info
                node = node.right
            else:
                node.left_info -= info
                node = node.left

        if z == TNULL:
            print("Couldn't find key in the tree")
            return

        y = z
        y_color = y.color
        if z.left == TNULL:
            # no left child
            x = z.right
            self.__rb_transplant(z, z.right)
        elif (z.right == TNULL):
            # no right child
            x = z.left
            self.__rb_transplant(z, z.left)
        else:
            # two children
            y = self.minimum(z.right)
            y_color = y.color
            x = y.right
            if y.parent == z:
                x.parent = y
            else:
                self.__rb_transplant(y, y.right)
                y.right = z.right
                y.right.parent = y

                # update info on path from y to x
                node = y.right
                key = x.data
                while node.data != key:
                    node.left_info -= y.info
                    node = node.left

            self.__rb_transplant(z, y)
            y.left = z.left
            y.blackheight = z.blackheight
            y.left.parent = y
            y.color = z.color
            y.left_info = z.left_info
            y.right_info = z.right_info - y.info
        if y_color == 0:
            self.__fix_delete(x)

    # fix the red-black tree
    def __fix_insert(self, k):
        while k.parent.color == 1:
            if k.parent == k.parent.parent.right:
                u = k.parent.parent.left  # uncle
                if u.color == 1:
                    # case 3.1
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1

                    u.blackheight += 1
                    k.parent.blackheight += 1

                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        # case 3.2.2
                        k = k.parent
                        self.right_rotate(k)
                    # case 3.2.1
                    k.parent.color = 0
                    k.parent.blackheight += 1
                    k.parent.parent.color = 1
                    k.parent.parent.blackheight -= 1
                    self.left_rotate(k.parent.parent)
            else:
                u = k.parent.parent.right  # uncle

                if u.color == 1:
                    # mirror case 3.1
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1

                    u.blackheight += 1
                    k.parent.blackheight += 1

                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        # mirror case 3.2.2
                        k = k.parent
                        self.left_rotate(k)
                    # mirror case 3.2.1
                    k.parent.color = 0
                    k.parent.blackheight += 1
                    k.parent.parent.color = 1
                    k.parent.parent.blackheight -= 1
                    self.right_rotate(k.parent.parent)
            if k == self.root:
                break
        if self.root.color == 1:
            self.root.color = 0
            self.root.blackheight += 1

    def __print_helper(self, node, indent, last,
                       color=False, infotoprint=False):
        # print the tree structure on the screen
        if node != TNULL:
            sys.stdout.write(indent)
            if last:
                sys.stdout.write("R----")
                indent += "     "
            else:
                sys.stdout.write("L----")
                indent += "|    "

            if color:
                if node.color == 1:
                    nstring = colored(str(node.data), 'red')
                else:
                    nstring = str(node.data)
            else:
                s_color = "RED" if node.color == 1 else "BLACK"
                nstring = str(node.data) + "(" + s_color + ")"
            if not(infotoprint):
                print(nstring)
            elif infotoprint == "right":
                print(nstring+" ({}) ({})".format(
                    node.info, node.right_info))
            elif infotoprint == "left":
                print(nstring+"({}) ({})".format(
                    node.info, node.left_info))
            elif infotoprint == "blackheight":
                print(nstring + "({})".format(node.blackheight))
            else:
                print(nstring+"({}) ".format(node.info))
            self.__print_helper(node.left, indent, False, color, infotoprint)
            self.__print_helper(node.right, indent, True, color, infotoprint)

    def __height(self, node):
        if node == TNULL:
            return 0
        else:
            return (1+max(self.__height(node.left), self.__height(node.right)))

    def __count(self, node):
        if node == TNULL:
            return 0
        else:
            return (1+self.__count(node.left) + self.__count(node.right))

    def __is_red_black_helper(self, node):
        """
        intermediate function to check that the subtree starting 
        at node respects the red black tree conditions.
        it returns a tuple. The first coordinate is True if it respects red black tree conditions.
        The second is the blackheight of this subtree
        """
        # leaf is black
        if node == TNULL:
            return (node.color == 0), 0

        rbl, bhl = self.__is_red_black_helper(node.left)
        rbr, bhr = self.__is_red_black_helper(node.right)
        if node.left == TNULL and node.right == TNULL:
            ordered = True
        elif node.left == TNULL:
            ordered = (node.right.data >= node.data)
        elif node.right == TNULL:
            ordered = (node.left.data <= node.data)
        else:
            ordered = (node.left.data <= node.data) and (node.right.data >= node.data)
        if node.color==1:
            #children must be black
            red = (node.left.color==0 and node.right.color==0)
        else:
            red = True
        # 1-node.color+max(bhl, bhr) is the blackheight
        # red indicates that in case of red node, its children are black
        # rbl indicates the left subtree respects red black tree conditions
        # rbr same with right subtree
        # bhl==bhr indicates that the subtree is balanced
        # ordered whether the tree is monotonically ordered (ie is a BST)
        return (red and rbl and rbr and (bhl==bhr) and ordered), 1-node.color+max(bhl, bhr)

    # Pre-Order traversal
    # Node.Left Subtree.Right Subtree
    def preorder(self):
        self.__pre_order_helper(self.root)

    # In-Order traversal
    # left Subtree . Node . Right Subtree
    def inorder(self):
        self.__in_order_helper(self.root)

    # Post-Order traversal
    # Left Subtree . Right Subtree . Node
    def postorder(self):
        self.__post_order_helper(self.root)

    # search the tree for the key k
    # and return the corresponding node
    def searchTree(self, k):
        return self.__search_tree_helper(self.root, k)

    # find the node with the minimum key
    def minimum(self, node):
        while node.left != TNULL:
            node = node.left
        return node

    # find the node with the maximum key
    def maximum(self, node):
        while node.right != TNULL:
            node = node.right
        return node

    # find the successor of a given node
    def successor(self, x):
        # if the right subtree is not None,
        # the successor is the leftmost node in the
        # right subtree
        if x.right != TNULL:
            return self.minimum(x.right)

        # else it is the lowest ancestor of x whose
        # left child is also an ancestor of x.
        y = x.parent
        while y != TNULL and x == y.right:
            x = y
            y = y.parent
        return y

    # find the predecessor of a given node
    def predecessor(self,  x):
        # if the left subtree is not None,
        # the predecessor is the rightmost node in the
        # left subtree
        if (x.left != TNULL):
            return self.maximum(x.left)

        y = x.parent
        while y != TNULL and x == y.left:
            x = y
            y = y.parent

        return y

    # rotate left at node x
    def left_rotate(self, x):
        y = x.right

        # change beta
        x.right = y.left
        if y.left != TNULL:
            y.left.parent = x

        # exchange x and y
        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        # augmented trees
        y.left_info += x.info + x.left_info
        x.right_info -= y.info + y.right_info

    # rotate right at node x
    def right_rotate(self, y):
        x = y.left
        # change beta
        y.left = x.right
        if x.right != TNULL:
            x.right.parent = y

        x.parent = y.parent
        if y.parent == None:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        x.right = y
        y.parent = x
        x.right_info += y.info + y.right_info
        y.left_info -= x.info + x.left_info

    # insert the key to the tree in its appropriate position
    # and fix the tree
    def insert(self, key, info):
        # Ordinary Binary Search Insertion
        node = Node(key, info)
        node.parent = None
        node.data = key
        node.left = TNULL
        node.right = TNULL
        node.color = 1  # new node must be red

        y = None
        x = self.root

        while x != TNULL:
            y = x
            if node.data <= x.data:
                x.left_info += info
                x = x.left
            else:
                x.right_info += info
                x = x.right

        # y is parent of x
        node.parent = y
        # if new node is a root node, simply return
        if y == None:
            self.root = node
            node.color = 0
            node.blackheight = 1
            return
        elif node.data <= y.data:
            y.left = node
        else:
            y.right = node

        # if the grandparent is None, simply return
        if node.parent.parent == None:
            return

        # Fix the tree
        self.__fix_insert(node)

    def join(self, T, k, info):
        """
        join with a second tree such that all keys in self are < k 
        and all key in T are > k
        (and add a node (k, info))
        """
        # Ordinary Binary Search Insertion
        node = Node(k, info)
        node.parent = None
        node.data = k
        node.info = info
        node.left = TNULL
        node.right = TNULL
        node.color = 1  # new node must be red

        if self.root.blackheight >= T.root.blackheight:
            Tfull = T.root.info + T.root.left_info + T.root.right_info
            x = self.root
            y = None
            while (x.blackheight > T.root.blackheight or x.color == 1):
                y = x
                x.right_info += Tfull + node.info
                x = x.right

            node.parent = y
            if y != None:
                y.right = node
            node.left = x
            node.right = T.root
            node.blackheight = x.blackheight
            x.parent = node
            T.root.parent = node
            node.left_info = x.info + x.right_info + x.left_info
            node.right_info = Tfull

        else:
            # mirrored
            selfull = self.root.info + self.root.left_info + self.root.right_info
            x = T.root
            y = None
            while (x.blackheight > self.root.blackheight or x.color == 1):
                y = x
                x.left_info += selfull + node.info
                x = x.left

            node.parent = y
            if y != None:
                y.left = node
            node.left = self.root
            node.right = x
            node.blackheight = x.blackheight
            x.parent = node
            self.root.parent = node
            node.left_info = selfull
            node.right_info = x.info + x.right_info + x.left_info
            self.root = T.root

        if y == None:
            node.color = 0
            node.blackheight += 1
            self.root = node

        elif node.parent.parent != None:
            # Fix the tree
            self.__fix_insert(node)

    def deletebelow(self, key):
        """
        delete all nodes < key
        Adapted from split operation but we only need to reconstruct
        one of the two trees here
        """
        # find smallest node with data > key
        x = self.root
        y = None
        R = []
        l = 0
        while x != TNULL:
            y = x
            if x.data >= key:
                T = RedBlackTree()
                T.root = x.right
                R.append((T, x.data, x.info))
                x = x.left
            else:
                x = x.right
                l += 1

        if len(R) > 0 and l > 0:
            # join all trees in R
            R[-1][0].root.parent = None
            if R[-1][0].root.color == 1:
                R[-1][0].root.color = 0
                R[-1][0].root.blackheight += 1

            R[-1][0].insert(R[-1][1], R[-1][2])
            for i in reversed(range(1, len(R))):
                R[i-1][0].root.parent = None
                if R[i-1][0].root.color == 1:
                    R[i-1][0].root.color = 0
                    R[i-1][0].root.blackheight += 1

                R[-1][0].join(R[i-1][0], R[i-1][1], R[i-1][2])
            self.root = R[-1][0].root

    def deleteabove(self, key):
        """
        delete all nodes > key (adapted from split operation)
        and return sum of info of all deleted nodes (useful for computation of phi_n)
        """
        # find smallest node with data > key
        x = self.root
        L = []
        infosum = np.zeros_like(x.info)
        r = 0
        while x != TNULL:
            y = x
            if x.data > key:
                infosum += x.info + x.right_info
                x = x.left
                r += 1
            else:
                T = RedBlackTree()
                T.root = x.left
                L.append((T, x.data, x.info))
                x = x.right

        if len(L) > 0 and r > 0:
            # join all trees in L
            L[-1][0].root.parent = None
            if L[-1][0].root.color == 1:
                L[-1][0].root.color = 0
                L[-1][0].root.blackheight += 1
            L[-1][0].insert(L[-1][1], L[-1][2])
            for i in reversed(range(1, len(L))):
                L[i-1][0].root.parent = None
                if L[i-1][0].root.color == 1:
                    L[i-1][0].root.color = 0
                    L[i-1][0].root.blackheight += 1
                L[i-1][0].join(L[i][0], L[i-1][1], L[i-1][2])

            self.root = L[0][0].root

        return infosum

    def get_root(self):
        return self.root

    def height(self):
        return self.__height(self.root)

    def count(self):
        return self.__count(self.root)

    # delete the node from the tree
    def delete_node(self, key, info):
        self.__delete_node_helper(self.root, key, info)

    def is_red_black(self):
        """
        check if the tree is indeed a red black tree
        """
        return(self.root.color == 0 and self.__is_red_black_helper(self.root)[0])

    # print the tree structure on the screen
    def pretty_print(self, color=False, infotoprint=False):
        """
        color: if True, print with colors (with notebooks only)
        infotoprint: False -> print only node keys
             True -> print keys and info
             'left' -> print keys, info and left_info
             'right' -> print keys, info and right_info
             'blackheight' -> print black height
        """
        self.__print_helper(self.root, "", True, color, infotoprint)


if __name__ == "__main__":
    pass
