import sys
sys.path.append("..")
import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet

max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# x.shape = (300, 2), t.shape = (300, 3)
x, t = spiral.load_data()
model = TwoLayerNet(input_size = 2, hidden_size = hidden_size, output_size = 3)
optimizer = SGD(lr = learning_rate)

data_size = len(x)
# 10 = 300 // 30
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

# epoch = 0 ... 300
for epoch in range(max_epoch):
    # データのシャッフル
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    # iters = 0, ... , 9
    for iters in range(max_iters):

        # data xとtのslicing [0:30], [30:60],...
        batch_x = x[iters*batch_size : (iters+1)*batch_size]
        batch_t = t[iters*batch_size : (iters+1)*batch_size]

        # 勾配を求め、パラメータを更新
        loss = model.forward(batch_x, batch_t)
        # 各レイヤのbackward()の過程で, 各々のself.gradsが更新される
        model.backward()
        # 各レイヤのparamsが更新される
        # params[i] -= self.lr * grads[i]
        optimizer.update(model.params, model.grads)

        total_loss = loss
        loss_count += 1

        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print()
