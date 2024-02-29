import string
import unicodedata
from collections import deque
from io import open
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def findFiles(path): return glob.glob(path)


all_letters = string.ascii_letters
n_letters = len(all_letters)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )


category_lines = {}
all_categories = []


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

import torch


def letterToIndex(letter):
    return all_letters.find(letter)


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

        self.h2o = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128

rnn = RNN(n_letters, n_hidden, n_categories)


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


import random


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)

    return category, line, category_tensor, line_tensor


criterion = nn.NLLLoss()

learning_rate = 0.01


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


import time
import math

n_iters: int = 1_000_000
prints_every = 10_000
plot_every = 100

all_losses = deque(maxlen=200)
all_accuracies = deque(maxlen=200)
correct = 0
runs = 0
accuracy = 0

fig, axs = plt.subplots(2, 1)
accuracy_ax = axs[0]
loss_ax = axs[1]


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    guess, guess_i = categoryFromOutput(output)

    # Reset accuracy all hundred runs
    runs += 1

    if runs % 200 == 0:
        accuracy = (correct / runs) * 100
        correct = 0
        runs = 0

    if guess == category:
        correct += 1


    # Print ``iter`` number, loss, name and guess
    if iter % prints_every == 0:
        correct_str = '✓' if guess == category else '✗ (%s)' % category
        print(
            '%d %d%% (%s) %.2f%% %.4f %s / %s %s' % (
                iter, iter / n_iters * 100, timeSince(start), accuracy, loss, line, guess, correct_str))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_accuracies.append(accuracy)
        all_losses.append(loss)
        accuracy_ax.clear()  # Clear the previous plot
        loss_ax.clear()
        accuracy_ax.plot(all_accuracies, color='blue')  # Plot the accuracy data
        loss_ax.plot(all_losses, color='blue')
        # Fit a linear trend line to the data
        x = np.arange(len(all_accuracies))
        slope, intercept, _, _, _ = stats.linregress(x, all_accuracies)
        trend_line = intercept + slope * x

        # Plot the trend line
        accuracy_ax.plot(trend_line, color='red', linestyle='--')

        accuracy_ax.grid(True)
        plt.pause(0.01)
