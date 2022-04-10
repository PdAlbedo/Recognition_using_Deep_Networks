"""
Take various experimentation with the deep network for the MNIST task
"""
__author__ = "Xichen Liu"

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from matplotlib import pyplot as plt
import deepNetwork

# set the random seed
torch.manual_seed(888)
cudnn.enabled = False

BATCH_SIZE_TRAIN = [32, 64, 128]
BATCH_SIZE_TEST = 1000
EPOCHS = [5, 7, 9]
LEARNING_RATE = 0.05
MOMENTUM = 0.5
CONV_FILTER_SIZE = [3, 5, 7]
DROPOUT_RATE = [0.3, 0.5, 0.7]
LOG_INTERVAL = 10


class NetworkVariation(nn.Module):
    def __init__(self, conv_filter_size, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, (conv_filter_size, conv_filter_size))
        self.conv2 = nn.Conv2d(10, 20, (conv_filter_size, conv_filter_size))
        self.conv2_drop = nn.Dropout2d(dropout_rate)
        self.flat1 = nn.Flatten()
        outer = conv_filter_size // 2
        size = ((28 - 2 * outer) // 2 - 2 * outer) // 2
        self.fc1 = nn.Linear(20 * size * size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # A convolution layer with 10 5x5 filters
        x = self.conv1(x)
        # A max pooling layer with a 2x2 window and a ReLU function applied
        x = F.relu(F.max_pool2d(x, (2, 2)))
        # A convolution layer with 20 5x5 filters
        x = self.conv2(x)
        # A dropout layer with a 0.5 dropout rate (50%)
        x = self.conv2_drop(x)
        # A max pooling layer with a 2x2 window and a ReLU function applied
        x = F.relu(F.max_pool2d(x, (2, 2)))
        # A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function on the
        # output
        x = F.relu(self.fc1(self.flat1(x)))
        # A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output
        x = F.log_softmax(self.fc2(x), dim = 1)

        return x


def get_loader(batch_size_train):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_mnist = datasets.MNIST('mnist',
                                 train = True,
                                 download = False,
                                 transform = transform
                                 )
    test_mnist = datasets.MNIST('mnist',
                                train = False,
                                download = False,
                                transform = transform
                                )
    train_loader = DataLoader(dataset = train_mnist,
                              batch_size = batch_size_train,
                              shuffle = True,
                              num_workers = 4)
    test_loader = DataLoader(dataset = test_mnist,
                             batch_size = BATCH_SIZE_TEST,
                             shuffle = True,
                             num_workers = 4)

    return train_loader, test_loader


def train_and_test(train_loader, test_loader, batch_size, epochs, conv_filter_size, dropout_rate,
                   learning_rate, momentum, log_interval):
    """
    train the model on training set epochs# times

    :param train_loader:    dataloader of training set
    :param test_loader:     dataloader of testing set
    :param batch_size:      batch_size of training
    :param epochs:          times complete pass through the training data
    :param conv_filter_size:size of conv filter
    :param dropout_rate:    dropout rate
    :param learning_rate:   learning rate
    :param momentum:        refers to inertia
    :param log_interval:    interval between every two recordings of loss
    """
    # initialize model and optimizer
    network = NetworkVariation(conv_filter_size, dropout_rate)
    optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum = momentum)
    train_losses = []
    train_counter = []
    test_losses_training_set = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(epochs + 1)]

    deepNetwork.test_network(network, train_loader, test_losses_training_set, 0, False)
    deepNetwork.test_network(network, test_loader, test_losses, 0, True)
    for epoch in range(1, epochs + 1):
        deepNetwork.train_network(network, train_loader, epoch, optimizer, log_interval, train_losses,
                                  train_counter, False)
        deepNetwork.test_network(network, train_loader, test_losses_training_set, epoch, False)
        deepNetwork.test_network(network, test_loader, test_losses, epoch, True)

    # torch.save(network.state_dict(), 'model.pt')
    # torch.save(optimizer.state_dict(), 'optimizer.pt')

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color = 'blue')
    plt.scatter(test_counter, test_losses_training_set, color = 'green', alpha = 1)
    plt.scatter(test_counter, test_losses, color = 'red', alpha = 1)
    plt.legend(['Train Loss', 'Test Loss on training set', 'Test Loss on test set'], loc = 'upper right')
    plt.title('Batch size: {} Epoch #: {} Conv filter size: {} Dropout rate: {}'.format(batch_size,
                                                                                        epochs,
                                                                                        conv_filter_size,
                                                                                        dropout_rate))
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig('designedNNPlt/Bsz_{}_E_{}_Fsz_{}_Drt_p{}.png'.format(batch_size,
                                                                      epochs,
                                                                      conv_filter_size,
                                                                      int(dropout_rate * 10)))
    fig.show()


def main():
    c = 1
    for i in BATCH_SIZE_TRAIN:
        train_loader, test_loader = get_loader(i)
        for j in EPOCHS:
            for p in CONV_FILTER_SIZE:
                for q in DROPOUT_RATE:
                    start = time.time()
                    print('########################################################################')
                    print('Training and evaluating for variation %d:' % c)
                    print('\tBatch size: {}'.format(i))
                    print('\tEpoch #: {}'.format(j))
                    print('\tConv filter size: {}'.format(p))
                    print('\tDropout rate: {}\n'.format(q))
                    train_and_test(train_loader, test_loader, i, j, p, q, LEARNING_RATE, MOMENTUM, LOG_INTERVAL)
                    end = time.time()
                    print("Time cost: %.2fs" % (end - start))
                    print('########################################################################\n')
                    c += 1


if __name__ == "__main__":
    main()
