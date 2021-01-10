import numpy as np


class Layer:
    def __init__(self, pre_num, num, activation):
        # pre_num是输入矩阵的维度, num是输出矩阵的维度
        self.num = num
        self.pre_num = pre_num
        # 储存layer在forward中通过激活函数的输出
        self.activation = activation
        self.parameters_init()
        self.activate_output = None
        self.dw = None
        self.db = None
        
    def activate(self, input):
        if self.activation == 'Relu':
            output = (np.abs(input) + input) / 2
            return output

        if self.activation == 'Sigmoid':
            output = 1 / (1 + np.exp(-input))
            return output

        if self.activation == 'Tanh':
            output = (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
            return output

        if self.activation == 'None':
            output = input
            return output

    def activate_derivative(self, input):
        if self.activation == 'Relu':
            output = np.array(input, copy=True)
            output[input > 0] = 1.
            output[input <= 0] = 0.
            return output

        elif self.activation == 'Sigmoid':
            output = np.multiply(input, (1.0 - input))
            return output

        elif self.activation == 'Tanh':
            output = 1.0 - input ** 2
            return output

        elif self.activation == 'None':
            output = np.ones_like(input)
            return output

    def forward(self, input):
        y = np.matmul(input, self.weight) + self.bias
        y = self.activate(y)
        self.activate_output = y
        return y

    def backpropagation(self, dz, pre_output, batch_size):
        # 根据误差进行反向传播，计算dw和db
        self.dw = np.dot(pre_output.T, dz) / batch_size
        self.db = np.mean(dz, axis=0, keepdims=True)

    def parameters_init(self):
        # 权重初始化，这里使用高斯分布随机初始化
        self.weight = np.random.randn(self.pre_num, self.num) / np.sqrt(self.pre_num)
        self.bias = np.zeros([1, self.num])

    def update_parameters(self, lr):
        self.weight = self.weight + lr * self.dw
        self.bias = self.bias + lr * self.db


class Network:
    def __init__(self, input_size, batch_size, loss, learning_rate):
        self.layers = []
        self.layer_num = 0
        self.input_size = input_size
        # 代表输入的向量维度
        self.pre_num = input_size
        self.batch_size = batch_size
        self.loss = loss
        self.learning_rate = learning_rate

    def add_layer(self, num, activation):
        layer = Layer(self.pre_num, num, activation)
        self.layers.append(layer)
        self.layer_num = self.layer_num + 1
        self.pre_num = num

    def count_parameters(self):
        parameters = 0
        for layer in self.layers:
            parameters = parameters + layer.pre_num * layer.num + layer.num
        return parameters

    def forward(self, input):
        output = None
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return output
        
    def backpropagation(self, input, y):
        output = self.forward(input)
        loss = self.get_loss(output, y)
        dz = self.get_loss_derivative(output, y)
        for i in range(self.layer_num-1, -1, -1):
            # 如果一共有三层（不计算输入层）那么i = 2, 1, 0
            layer = self.layers[i]
            if i == 0:
                pre_output = input
                # 如果是输入层，那么这一层的输入为网络的输入
            else:
                pre_output = self.layers[i - 1].activate_output
                # 如果不是输入层，那么这一层的输入为上一层的输出
            layer.backpropagation(dz, pre_output, self.batch_size)
            if i != 0:
                dh = np.dot(dz, layer.weight.T)
                dz = np.multiply(dh, self.layers[i - 1].activate_derivative(self.layers[i - 1].activate_output))
        return loss

    def update_parameters(self, lr):
        for i in range(self.layer_num):
            layer = self.layers[i]
            layer.update_parameters(lr)

    def get_loss(self, output, y):
        loss = None
        if self.loss == 'mse':
            loss = np.nanmean((y-output)**2)
        elif self.loss == 'mae':
            loss = np.nanmean(np.absolute(y-output))
        elif self.loss == 'logprobs':
            output_exp = np.exp(output)
            output_prob = output_exp / np.sum(output_exp, axis = -1, keepdims = True)
            loss = np.mean(np.sum(-y * np.log(output_prob), axis = -1))
        loss = loss.reshape(1, 1)
        return loss
        
    def get_loss_derivative(self, output, y):
        loss_derivative = None
        if self.loss == 'mse':
            loss_derivative = y - output
        elif self.loss == 'mae':
            loss_derivative = y - output
            loss_derivative[loss_derivative >= 0] = 1
            loss_derivative[loss_derivative < 0] = -1
        elif self.loss == 'logprobs':
            loss_derivative = y - output
        return loss_derivative
