import sys
import numpy as np
from nn_2 import SGD, TwoLayerNet
sys.path.append('./deep-learning-from-scratch-2/')
from dataset import spiral
import matplotlib.pyplot as plt

# Hyper Parameter
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# DataLoad
x, t = spiral.load_data()
# Model Setup
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

# Variables
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    # data shuffle
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    # ここが1つの学習セットにあたる
    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]

        # Gradient
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        # verbose
        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print(f"| epoch {epoch+1} | iter {iters+1} / {max_iters} | loss {avg_loss}.2f")
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0

