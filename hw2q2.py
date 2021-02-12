import random
import numpy as np
import matplotlib.pyplot as plt


class Value:
    """
    Basic unit of storing a single scalar value and its gradient
    """

    def __init__(self, data, _children=()):
        """

        """
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        """
        Example implementation of a single class operation (addition)

        Args:
            other (Any): Node to add with the class

        Returns:
            out (callable): Function to referesh the gradient
        """
        # Firstly, convert some default value type in python to Value
        # Then do operations with two or more Value object
        other = other if isinstance(other, Value) else Value(other)

        # Secondly, create a new Value object which is the result of the operation
        out = Value(self.data + other.data, (self, other))

        # Thirdly, create a _backward function for the output object to refresh
        # the gradient of its _childrens,
        # Then assign this _backward function to the output object.
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        Multiplication operation (e.g. Value(3) * Value(2) = Value(6))
        """
        # TODO implement multiplication operation

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __pow__(self, other):
        """
        Power operation (e.g Value(3) ** 2 = Value(8))
        """
        # TODO implement multiplication operation, we don't need to convert the exponent to Value
        assert isinstance(other, (int, float))

        out = Value(self.data ** other, (self, ))

        def _backward():
            self.grad += out.grad * other * (self.data ** (other - 1))

        out._backward = _backward
        return out

    def relu(self):
        """
        ReLU activation function applied to the current Value
        """
        # TODO implement the relu activation function for the value itself.
        # equivalent to implement max(0, self.data)
        if self.data < 0:
            out = Value(0, (self, ))
        else:
            out = Value(self.data, (self, ))

        def _backward():
            if self.data < 0:
                self.grad += 0
            else:
                self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def exp(self):
        """
        Exponentiate the current Value (e.g. e ^ Value(0) = Value(1))
        """
        # TODO implement the exponential function for and treat the value as exponent.
        # The base is natural e, you can use numpy to calculate the value of the exponential.
        out = Value(np.exp(self.data), (self, ))

        def _backward():
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def log(self):
        """
        Take the natural logarithm (base e) of the current Value
        """
        # TODO implement the logarithm function for and treat the value as exponent.
        # The bottom number should be e, you can use numpy to calculate the value of the logarithm.
        out = Value(np.log(self.data), (self, ))

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def dfs(self, nodeSet, topoList):
        # Implement the Deep First Search for a graph
        # Used in the backward() function for topological sort
        if self not in nodeSet:
            nodeSet.add(self)
            topoList.append(self)
            for item in self._prev:
                item.dfs(nodeSet, topoList)

    def backward(self):
        """
        Run backpropagation from the current Value
        """
        # This function is called when you start backpropagation from this Value

        # The gradient of this value is initialized to 1 for you.
        self.grad = 1

        # You need to find a right topological order all of the children in the graph.
        # As for topology sort, you can refer to http://www.cs.cornell.edu/courses/cs312/2004fa/lectures/lecture15.htm

        topo = []
        # TODO find the right list of Value to be traversed
        '''
        Hint: you can recursively visit all non-visited node from the node calling backward.
        add one node to the head of the list after all of its children node are visited
        '''
        nodeSet = set()
        self.dfs(nodeSet, topo)
        # go one variable at a time and apply the chain rule to get its gradient

        for v in topo:
            v._backward()

    # We handled the negation and reverse operations for you
    def __neg__(self):  # -self
        """
        Negate the current Value
        """
        return self * -1

    def __radd__(self, other):  # other + self
        """
        Reverse addition operation (ordering matters in Python)
        """
        return self + other

    def __sub__(self, other):  # self - other
        """
        Subtraction operation
        """
        return self + (-other)

    def __rsub__(self, other):  # other - self
        """
        Reverse subtraction operation
        """
        return other + (-self)

    def __rmul__(self, other):  # other * self
        """
        Reverse multiplication operation
        """
        return self * other

    def __truediv__(self, other):  # self / other
        """
        Division operation
        """
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        """
        Reverse diction operation
        """
        return other * self ** -1

    def __repr__(self):
        """
        Class representation (instead of unfriendly memory address)
        """
        return f"Value(data={self.data}, grad={self.grad})"


w1 = Value(0.3)
w2 = Value(-0.5)
x1 = Value(0.2)
x2 = Value(0.4)

# TODO
# Do calculation for the question 1.b, and call backward to start backpropagation.
# Then print out the gradient of w1 w2 x1 x2.

v1 = w1.__mul__(x1)   # w1x1
v2 = w2.__mul__(x2)   # w2x2
v3 = v1.__add__(v2)   # w1x1 + w2x2
v4 = v3.__mul__(-1)   # -(w1x1 + w2x2)
v5 = v4.exp()         # e^(-(w1x1 + w2x2))
v6 = v5.__add__(1)    # 1 + e^(-(w1x1 + w2x2))
v7 = v6.__pow__(-1)   # 1 / 1 + e^(-(w1x1 + w2x2))

u1 = w1.__pow__(2)    # w1^2
u2 = w2.__pow__(2)    # w2^2
u3 = u1.__add__(u2)   # w1^2 + w2^2
u4 = u3.__mul__(0.5)  # 0.5(w1^2 + w2^2)

f1 = v7.__add__(u4)   # (1 / 1 + e^(-(w1x1 + w2x2))) + (0.5(w1^2 + w2^2))

f1.backward()
print(w1.grad, w2.grad, x1.grad, x2.grad)
