import numpy as np
from common.functions import softmax, cross_entropy_error


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        x = self.x
        dx = np.dot(dout, W.T)
        dW = np.dot(x.T, dout)
        self.grads[0][...] = dW
        return dx

class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None

    def forward(self, x):
        self.out =  1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        out = self.out
        dx = dout * out * (1.0 - out)
        return dx

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        self.x = x
        out = np.dot(x, N) + b
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.out(dout, axis=0)
        self.grads[0][...] = dw
        self.grads[1][...] = db
        return dx

# 未完
# class SoftmawWithLoss:
#     def __init__(self):
#         self.params = []
#         self.grads = []
#         self.y = None
#         self.t = None
#
#     def forward(self, a, t):
#         self.t = t
#         # softmax layer
#         y = np.exp(a) / np.sum(np.exp(a))
#         self.y = y
#         loss = -1 * np.sum(self.t * y)
#         return loss
#
#     def backward(self):
#         y = self.y
#         t = self.t
#         return y - t

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx

#
#
# W = np.random.randn(3, 5)
# mm = MatMul(W)
# mm.forward(np.random.randn(7, 3))
# print(mm.backward(np.random.randn(7, 5)))
