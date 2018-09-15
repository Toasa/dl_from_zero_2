import sys
sys.path.append("..")
import numpy as np
from common.layers import MatMul

# You see the moon and I see the sun .
# you see the moon and I sun .

c0 = np.array([[0, 0, 1, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 0, 0, 1, 0, 0, 0]])

W_in = np.random.randn(8, 4)
W_out = np.random.randn(4, 8)

in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
print(h1.shape)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)

print(s.shape)
