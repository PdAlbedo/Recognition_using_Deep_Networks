"""
Project 5
Recognition using Deep Networks
"""
__author__ = "Xichen Liu"

import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from matplotlib import pyplot as plt

BATCH_SIZE_TRAIN = 64  # batch size for training and testing
BATCH_SIZE_TEST = 1000
EPOCHS = 5  # iterations of training
LEARNING_RATE = 0.01  # learning rate
MOMENTUM = 0.5  # refers to inertia
LOG_INTERVAL = 10  # record the loss for every interval# batches passed


# class definitions
class MyNetwork(nn.Module):
    def __init__(self, conv_filter = 5, dropout_rate = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, (conv_filter, conv_filter))
        self.conv2 = nn.Conv2d(10, 20, (conv_filter, conv_filter))
        self.conv2_drop = nn.Dropout2d(dropout_rate)
        # flatten operation, which is equal to .view(-1, 320)
        self.flat1 = nn.Flatten()
        outer = conv_filter // 2
        size = ((28 - 2 * outer) // 2 - 2 * outer) // 2
        self.fc1 = nn.Linear(20 * size * size, 50)
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    # methods need a summary comment
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


# # For extension
# class MyNetwork(nn.Module):
#     def __init__(self, kernels):
#         super().__init__()
#         self.kernels = kernels
#         self.conv1 = nn.Conv2d(1, 10, (5, 5))
#         self.conv2 = nn.Conv2d(10, 20, (5, 5))
#         self.conv2_drop = nn.Dropout2d(0.5)
#         self.flat1 = nn.Flatten()
#         self.fc1 = nn.Linear(20 * 4 * 4, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         imgs = []
#         for i in range(len(x)):
#             kernels = []
#             for j in self.kernels:
#                 img = cv2.filter2D(x[i][0].detach().numpy(), -1, j, borderType = None)
#                 H = np.floor(np.array(j.shape) / 2).astype(np.int64)
#                 valid_img = img[H[0]:-H[0], H[1]:-H[1]]
#                 kernels.append(valid_img)
#             imgs.append(kernels)
#         x = torch.from_numpy(np.array(imgs))
#         # x = x[None, :]
#         # A max pooling layer with a 2x2 window and a ReLU function applied
#         x = F.relu(F.max_pool2d(x, (2, 2)))
#         # A convolution layer with 20 5x5 filters
#         x = self.conv2(x)
#         # A dropout layer with a 0.5 dropout rate (50%)
#         x = self.conv2_drop(x)
#         # A max pooling layer with a 2x2 window and a ReLU function applied
#         x = F.relu(F.max_pool2d(x, (2, 2)))
#         # A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function on the
#         # output
#         x = F.relu(self.fc1(self.flat1(x)))
#         # A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output
#         x = F.log_softmax(self.fc2(x), dim = 1)
#         return x

def read_and_print(batch_size_train, batch_size_test, is_show):
    """
    return the dataset read and plot the 1st 6 digits.

    :param batch_size_train:    batch size for training
    :param batch_size_test:     batch size for testing
    :param is_show:             determine whether show the 1st 6 digits
    :return the train dataloader and test_network dataloader
    """
    # load the mnist dateset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_mnist = datasets.MNIST('mnist',
                                 train = True,
                                 download = True,
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
                             batch_size = batch_size_test,
                             shuffle = True,
                             num_workers = 4)

    # extract and store the date of each batches
    # elements in train_data and train_label are in type of torch.size()
    num_batch = enumerate(train_loader)
    # batch_idx = 0
    train_data = []
    train_label = []
    while True:
        try:
            batch_idx, (tmp_data, tmp_label) = next(num_batch)
            train_data.append(tmp_data)
            train_label.append(tmp_label)
        except StopIteration:
            break
    # print(train_data.__sizeof__())
    # print(train_label.__sizeof__())

    if is_show:
        # plot the first 6 digits
        fig = plt.figure()
        i = 0
        c = 0
        while i <= (6 // BATCH_SIZE_TEST):
            for j in range(BATCH_SIZE_TEST):
                if c >= 6:
                    i = float('inf')
                    break
                plt.subplot(2, 3, c + 1)
                plt.tight_layout()
                plt.imshow(train_data[i][j][0], cmap = 'gray', interpolation = 'none')
                plt.title("GroundTruth: %d" % train_label[i][j])
                plt.xticks([])
                plt.yticks([])
                c += 1
            i += 1
        fig.show()

    return train_loader, test_loader


def train_network(network, train_loader, epoch, optimizer, log_interval, train_losses, train_counter, show_process):
    """
    train every batch of the training dataloader

    :param network:         model of neural network
    :param train_loader:    dataloader of training data
    :param epoch:           times complete pass through the training data
    :param optimizer:       the function or algorithm to minimize the loss by adjusting weights or learning rate
    :param log_interval:    interval between every two recordings of loss
    :param train_losses:    value of loss at each value of training counter
    :param train_counter:   number of training example seen
    :param show_process:    determine whether the training process will be shown
    """
    # set network to train
    network.train()
    # loop over every batch
    # print('\n')
    for batch_idx, (data, label) in enumerate(train_loader):
        # set optimizer to 0 gradient at the beginning
        optimizer.zero_grad()
        # compute the output and calculate loss by cross entropy
        output = network(data)
        loss = F.cross_entropy(output, label)
        # Computes the gradient of current tensor w.r.t. graph leaves
        loss.backward()
        # update parameters
        optimizer.step()
        if batch_idx % log_interval == 0:
            if show_process:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))


def test_network(network, test_loader, test_losses, epoch, is_test_set):
    """
    evaluating the model on test set

    :param network:     model of neural network
    :param test_loader: dataloader of test set
    :param test_losses: value of losses at each point in test counter
    """
    # set model to eval()
    network.eval()
    test_loss = 0
    correct = 0
    # disable gradient calculation is useful for inference, backward() will not be called in testing
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.cross_entropy(output, target, reduction = 'sum').item()
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    if epoch == 0:
        if is_test_set:
            print('Initial test on test set: Avg.loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        else:
            print('Initial test on training set: Avg.loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
    else:
        if is_test_set:
            print('Test on test set after epoch {}: Avg.loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                epoch, test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        else:
            print('\nTest on training set after epoch {}: Avg.loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                epoch, test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))


def train_and_test(train_loader, test_loader, epochs, learning_rate, momentum, log_interval):
    """
    train the model on training set epochs# times

    :param train_loader:    dataloader of training set
    :param test_loader:     dataloader of testing set
    :param epochs:          times complete pass through the training data
    :param learning_rate:   learning rate
    :param momentum:        refers to inertia
    :param log_interval:    interval between every two recordings of loss
    """
    # initialize model and optimizer

    # # for extension
    # filters = []
    #
    # for theta in np.arange(0, np.pi, np.pi / 2):
    #     for k in range(5):
    #         kernel = cv2.getGaborKernel((5, 5), 1.0, theta, np.pi / 2.0, 0.5, 0, ktype = cv2.CV_32F)
    #         kernel /= 1.5 * kernel.sum()
    #         filters.append(kernel)
    # network = MyNetwork(filters)
    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum = momentum)
    train_losses = []
    train_counter = []
    test_losses_training_set = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(epochs + 1)]

    test_network(network, train_loader, test_losses_training_set, 0, False)
    test_network(network, test_loader, test_losses, 0, True)
    for epoch in range(1, epochs + 1):
        train_network(network, train_loader, epoch, optimizer, log_interval, train_losses, train_counter, True)
        test_network(network, train_loader, test_losses_training_set, epoch, False)
        test_network(network, test_loader, test_losses, epoch, True)

    torch.save(network.state_dict(), 'model.pt')
    torch.save(optimizer.state_dict(), 'optimizer.pt')

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color = 'blue')
    plt.scatter(test_counter, test_losses_training_set, color = 'green', alpha = 0.75)
    plt.scatter(test_counter, test_losses, color = 'red', alpha = 0.75)
    plt.legend(['Train Loss', 'Test Loss on training set', 'Test Loss on test set'], loc = 'upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig.show()


# main function
def main(argv):
    # handle any command line arguments in argv

    # main function code
    # set the random seed
    torch.manual_seed(888)
    cudnn.enabled = False

    # print the 1st 6 digits accordingly and return dataloader of training set and testing set
    train_loader, test_loader = read_and_print(BATCH_SIZE_TRAIN, BATCH_SIZE_TEST, True)
    # train and test the network epochs by epochs
    train_and_test(train_loader, test_loader, EPOCHS, LEARNING_RATE, MOMENTUM, LOG_INTERVAL)


if __name__ == "__main__":
    main(sys.argv)
