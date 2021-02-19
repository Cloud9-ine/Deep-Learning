import random
import numpy as np
import matplotlib.pyplot as plt


########################################################################################################################
# Question 2
########################################################################################################################
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
        # Implement the Depth First Search for a graph
        if self not in nodeSet:
            nodeSet.add(self)
            for item in self._prev:
                item.dfs(nodeSet, topoList)
            topoList.append(self)

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
        topo.reverse()
        # self.dfs(nodeSet, topo)
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
print('Question 2: gradient of w1, w2, x1, x2')
print('w1:', w1.grad, 'w2:', w2.grad, 'x1:', x1.grad, 'x2:', x2.grad)
print('\n')


########################################################################################################################
# Question 3
########################################################################################################################
class Module:
    """
    Base Model Module
    """

    def parameters(self):
        """

        """
        return []

    def zero_grad(self):
        """

        """
        for p in self.parameters():
            p.grad = 0


class LinearLayer(Module):
    """
    Linear Layer
    """

    def __init__(self, nin, nout):
        """
        Here we randomly initilize the weights w as 2-dimensional list of Values
        And b as 1-dimensional list of Values with value 0

        You may use this stucture to implement the __call__ function
        """
        self.w = []
        for i in range(nin):
            w_tmp = [Value(random.uniform(-1, 1)) for j in range(nout)]
            self.w.append(w_tmp)
        self.b = [Value(0) for i in range(nout)]
        self.nin = nin
        self.nout = nout

    def __call__(self, x):
        """
        Args:
            x (2d-list): Two dimensional list of Values with shape [batch_size , nin]

        Returns:
            xout (2d-list): Two dimensional list of Values with shape [batch_size, nout]
        """
        # TODO implement this function and return the output of a linear layer.
        batch_size = len(x)
        xout = []
        for i in range(batch_size):
            x_tmp = []
            for j in range(self.nout):
                v = Value(data=0)
                for k in range(self.nin):
                    v += self.w[k][j] * x[i][k]
                v += self.b[j]
                x_tmp.append(v)
            xout.append(x_tmp)
        return xout

    def parameters(self):
        """
        Get the list of parameters in the Linear Layer

        Args:
            None

        Returns:
            params (list): List of parameters in the layer
        """
        return [p for row in self.w for p in row] + [p for p in self.b]


# Initialization of Layer with Weights
linear_model_test = LinearLayer(4, 4)
linear_model_test.w = [
    [Value(data=0.7433570245252463), Value(data=-0.9662164096144394), Value(data=-0.17087204941322653),
     Value(data=-0.5186656374983067)],
    [Value(data=-0.1414882837892344), Value(data=-0.5898971049017006), Value(data=-0.3448340220492381),
     Value(data=0.5278833226346107)],
    [Value(data=0.3990701306597799), Value(data=-0.3319058654296163), Value(data=-0.784797384411202),
     Value(data=0.7603317495966846)],
    [Value(data=-0.5711035064293541), Value(data=-0.0001937643033362857), Value(data=0.12693226232877053),
     Value(data=-0.36044237239197097)]]
linear_model_test.b = [Value(data=0), Value(data=0), Value(data=0), Value(data=0)]

# Forward Pass
x_test = [[-0.17120438454836173, -0.3736077734087335, -0.48495413054653214, 0.8269206715993096]]
y_hat_test = linear_model_test(x_test)
y_ref = [[Value(data=-0.7401928625441141), Value(data=0.5466095223360173), Value(data=0.6436403600545564),
          Value(data=-0.7752067527386406)]]

# Error Calculation
predict_error = 0
for i in range(4):
    predict_error += (y_hat_test[0][i] - y_ref[0][i]) ** 2
print('Question 3: predict error')
print(predict_error.data)


# Implementation of Loss functions
def softmax(y_hat):
    """
    Softmax computation

    Args:
        y_hat (2d-list): 2-dimensional list of Values with shape [batch_size, n_class]

    Returns:
        s (2d-list): 2-dimensional list of Values with the same shape as y_hat
    """
    # TODO implement the softmax function and return the output.
    batch_size = len(y_hat)
    n_class = len(y_hat[0])
    s = []
    for i in range(batch_size):
        s_tmp = []
        exp_sum = Value(data=0)
        for j in range(n_class):
            exp_sum += y_hat[i][j].exp()
        for j in range(n_class):
            s_tmp.append(y_hat[i][j].exp() / exp_sum)
        s.append(s_tmp)
    return s


def cross_entropy_loss(y_hat, y):
    """
    Cross-entropy Loss computation

    Args:
        y_hat (2d-list): Output from linear function with shape [batch_size, n_class]
        y (1d-list): List of ground truth labels with shape [batch_size, ]

    Returns:
        loss (Value): Loss value of type Value
    """
    # TODO implement the calculation of cross_entropy_loss between y_hat and y.
    batch_size = len(y)
    n_class = len(y_hat[0])
    y_hat_soft = softmax(y_hat)
    loss = Value(data=0)
    for i in range(batch_size):
        for j in range(n_class):
            if y[i] == j:
                loss -= y_hat_soft[i][j].log()
    loss = loss / batch_size
    return loss


def accuracy(y_hat, y):
    """
    Accuracy computation

    Args:
        y_hat (2d-list): Output from linear function with shape [batch_size, n_class]
        y (1d-list): List of ground truth labels with shape [batch_size, ]

    Returns:
        acc (float): Accuracy score
    """
    # TODO implement the calculation of accuracy of the predicted y_hat w.r.t y.
    batch_size = len(y)
    n_class = len(y_hat[0])
    n_hit = 0
    for i in range(batch_size):
        tmp = y_hat[i][y[i]]
        flag = True
        for j in range(n_class):
            if tmp.data < y_hat[i][j].data:
                flag = False
                break
        if flag is True:
            n_hit += 1
    acc = n_hit / batch_size
    return acc


y_gt = [1]
y_hat_test = linear_model_test(x_test)
# print(y_hat_test)

# Softmax Calculation
prob_test = softmax(y_hat_test)
# print(prob_test)
prob_ref = [[0.10441739448437284, 0.37811510516540814, 0.4166428991676558, 0.10082460118256342]]
softmax_error = 0
for i in range(4):
    softmax_error += (prob_ref[0][i] - prob_test[0][i]) ** 2
print('Question 3: softmax error')
print(softmax_error.data)


# Cross Entropy Loss Calculation
loss_test = cross_entropy_loss(y_hat_test, y_gt)
loss_ref = Value(data=0.9725566186970217)
print('Question 3: cross entropy loss')
print((loss_test - loss_ref).data)


# Update Gradient Based on Loss
linear_model_test.zero_grad()
loss_test.backward()
w_gradient_ref = [[-0.017876715758840547, 0.10646942068007896, -0.07133109112844363, -0.01726161379279479],
                  [-0.0390111502584479, 0.23234103087567629, -0.1556610258645873, -0.03766885475264107],
                  [-0.05063764675610328, 0.30158564847453107, -0.2020526949142369, -0.04889530680419089],
                  [0.08634490197366762, -0.5142494748940867, 0.3445306259968013, 0.08337394692361787]]
b_gradient_ref = [0.10441739448437282, -0.6218848948345919, 0.4166428991676557, 0.1008246011825634]

# Compute Error
w_gradient_error = 0
b_gradient_error = 0
for i in range(4):
    b_gradient_error += (linear_model_test.b[i].grad - b_gradient_ref[i]) ** 2
    for j in range(4):
        w_gradient_error += (linear_model_test.w[i][j].grad - w_gradient_ref[i][j]) ** 2
print('Question 3: gradient error')
print(w_gradient_error)
print(b_gradient_error)
print('\n')

# print('w.grad:')
# for i in range(len(linear_model_test.w)):
#         print(linear_model_test.w[i][0].grad, linear_model_test.w[i][1].grad,
#               linear_model_test.w[i][2].grad, linear_model_test.w[i][3].grad)
# print('b.grad:')
# print(linear_model_test.b[0].grad, linear_model_test.b[1].grad,
#       linear_model_test.b[2].grad, linear_model_test.b[3].grad)


def plot_points(X, Y, scale, n, data):
    """
    Plot points in the visualization image
    """
    points_color = [[0., 0., 255.], [255., 0., 0.], [0., 255., 0.], [0., 0., 0.]]

    # TODO Assign a color to "data" according to the position and the label of X
    step = scale * 2 / n
    for i in range(X.shape[0]):
        x1 = int((X[i][0] + scale) / step)
        x2 = int((X[i][1] + scale) / step)
        label = int(Y[i])
        data[x1][x2][0] = points_color[label][0]
        data[x1][x2][1] = points_color[label][1]
        data[x1][x2][2] = points_color[label][2]
    return data


def plot_background(scale, n, model):
    """
    Color the background in the visualization image
    """

    background_color = [[0., 191., 255.], [255., 110., 180.], [202., 255., 112.], [156., 156., 156.]]

    data = np.zeros((n, n, 3), dtype='uint8')

    for i in range(n):
        x1 = -scale + 2 * scale / n * i
        for j in range(n):
            x2 = -scale + 2 * scale / n * j
            input = [[Value(x1), Value(x2)]]
            # TODO using the model to predict a class for the input and assign a color to "data" at this position.
            output = model(input)

            temp = output[0][0].data
            label = 0
            for k in range(1, len(output[0])):
                if output[0][k].data > temp:
                    temp = output[0][k].data
                    label = k

            data[i][j][0] = background_color[label][0]
            data[i][j][1] = background_color[label][1]
            data[i][j][2] = background_color[label][2]
    return data


def visualization(X, Y, model):
    """
    Decision boundary visualization
    """
    scale = 4.5  # the scale of X axis and Y axis. To say, x is from -scale to +scale
    n = 300  # seperate the image into n*n pixels

    data = plot_background(scale, n, model)
    data = plot_points(X, Y, scale, n, data)

    plt.imshow(data)
    plt.axis('off')
    plt.show()


def train(x,
          y,
          model,
          loss_function=cross_entropy_loss,
          accuracy_function=accuracy,
          max_iteration=500,
          learning_rate=1):
    """
    Args:
       x (2-d list): List of Values with shape: [n_samples, n_channels]
       y (1-d list): List of integers with shape: [n_samples]
       model (Module): Linear model
       loss_function (callable): Loss function to use during training
       accuracy_function (callable): Function used for calculating training accuracy
       max_iteration (int): Number of epochs to train model for
       learning_rate (numeric): Step size of the gradient update
    """
    for i in range(max_iteration):
        # TODO compute y_hat and calculate the loss between y_hat and y as well as
        # the accuracy of y_hat w.r.t y.
        y_hat = model(x)
        loss = loss_function(y_hat, y)
        acc = accuracy_function(y_hat, y)

        # TODO Then You will need to calculate gradient for all parameters, and
        # do gradient descent for all the parameters.
        # The list of parameters can be easily obtained by calling
        # mode.parameters() which is implemented above.
        loss.backward()
        paraList = model.parameters()
        for para in paraList:
            para.data -= learning_rate * para.grad
            para.grad = 0
        # print(paraList[0])

        # Then plot the loss / accuracy vs iterations.
        if i % 20 == 19:
            print("iteration", i, "loss:", loss.data, "accuracy:", acc)

        # record loss
        if i == 0:
            # initialize L
            L = loss.data
            A = acc
        else:
            L = np.append(L, loss.data)
            A = np.append(A, acc)

    ## Plot Loss and Accuracy
    fig0 = plt.figure(0)
    plt.plot(L, '-')
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Loss', fontsize=16)
    plt.show()
    fig1 = plt.figure(1)
    plt.plot(A, '-')
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Accuracy', fontsize=16)
    plt.show()


# Load Q3 Dataset
datapath = './Q3_data.npz'
data = np.load(datapath)

# Load Data and Parse Shape Information
X = data['X']
Y = data['Y']
print('Question 3: Linear')
print(X.shape, Y.shape, np.unique(Y))
print('\n')
nin = X.shape[1]
nout = np.max(Y) + 1

# # Initialize data using your Value class
x = [[Value(v) for v in sample] for sample in X]
y = [int(v) for v in Y]

# # Initialize a Linear Model
# linear_model = LinearLayer(nin, nout)
#
# # Train the Model using Your Data
# train(x, y, linear_model)
#
# # Visualize learned decision boundaries
# visualization(X, Y, linear_model)


########################################################################################################################
# Question 4
########################################################################################################################
# Load Q4 Dataset
datapath = './Q4_data.npz'
data = np.load(datapath)

# Parse Data and Identify Dimensions
X = data['X']
Y = data['Y']
print('Question 4: MLP')
print(X.shape, Y.shape, np.unique(Y))
print('\n')
nin = X.shape[1]
nout = int(np.max(Y)) + 1

# Initialize data using your value class
x = [[Value(v) for v in sample] for sample in X]
y = [int(v) for v in Y]

# Initialize Linear Model
linear_model = LinearLayer(nin, nout)

# Train Model
train(x, y, linear_model)

# Visualize Learned Decision Boundary
visualization(X, Y, linear_model)


class MLP(Module):
    """
    Multi Layer Perceptron
    """

    def __init__(self, dimensions):
        """
        Initialize multiple layers here in the list named self.linear_layers
        """
        assert isinstance(dimensions, list)
        assert len(dimensions) > 2
        self.linear_layers = []
        for i in range(len(dimensions) - 1):
            self.linear_layers.append(LinearLayer(dimensions[i], dimensions[i + 1]))

    def __call__(self, x):
        """
        Args:
            x (2d-list): Two dimensional list of Values with shape [batch_size , nin]

        Returns:
            xout (2d-list): Two dimensional list of Values with shape [batch_size, nout]
        """
        # TODO Implement this function and return the output of a MLP

        # first_out = self.linear_layers[0](x)
        # second_in = []
        # for i in range(len(first_out)):
        #     tmp = []
        #     for j in range(len(first_out[0])):
        #         v = Value(data=0)
        #         v = first_out[i][j].relu()
        #         tmp.append(v)
        #     second_in.append(tmp)
        # xout = self.linear_layers[1](second_in)
        # return xout

        outputs = [x]
        for L in range(len(self.linear_layers)):
            prev_out = self.linear_layers[L](outputs[-1])
            if L == len(self.linear_layers) - 1:
                return prev_out
            next_in = []
            for i in range(len(prev_out)):
                tmp = []
                for j in range(len(prev_out[0])):
                    v = Value(data=0)
                    v = prev_out[i][j].relu()
                    tmp.append(v)
                next_in.append(tmp)
            outputs.append(next_in)

    def parameters(self):
        """
        Get the parameters of each layer

        Args:
            None

        Returns:
            params (list of Values): Parameters of the MLP
        """
        return [p for layer in self.linear_layers for p in layer.parameters()]

    def zero_grad(self):
        """
        Zero out the gradient of each parameter
        """
        for p in self.parameters():
            p.grad = 0


# # Initialize MLP with Given Parameters
# mlp_model = MLP([nin, 40, nout])
#
# # Train the MLP
# train(x, y, mlp_model)
#
# # Visualize Decision Boundaries
# visualization(X, Y, mlp_model)
