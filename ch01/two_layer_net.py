import sys
sys.path.append("..")
import numpy as np
from common.layers import Affine, Sigmoid, SoftmawWithLoss

# Affine, sigmodの層とSoftmaxwithloss層の２つ
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        self.params, self.grad = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    # affine, sigmoid, affineと通過すれば、”推測”は終わる
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # 損失lossを得るため、SoftmawWithLoss層までforwardする
    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        # SoftmaxWithLoss層のbackward
        dout = self.loss_layer.backward(dout)

        # Affine, Sigmoid層のbackward
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
