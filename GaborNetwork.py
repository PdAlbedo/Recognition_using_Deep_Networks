"""
Try pre-trained networks available in the PyTorch package and evaluate its first couple of convolutional layers
"""
__author__ = "Xichen Liu"

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import deepNetwork


class NetworkGabor(nn.Module):
    def __init__(self, kernels):
        super().__init__()
        self.kernels = kernels
        self.conv1 = nn.Conv2d(1, 10, (5, 5))
        self.conv2 = nn.Conv2d(10, 20, (5, 5))
        self.conv2_drop = nn.Dropout2d(0.5)
        self.flat1 = nn.Flatten()
        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        imgs = []
        for i in range(len(x)):
            kernels = []
            for j in self.kernels:
                img = cv2.filter2D(x[i][0].detach().numpy(), -1, j, borderType = None)
                H = np.floor(np.array(j.shape) / 2).astype(np.int64)
                valid_img = img[H[0]:-H[0], H[1]:-H[1]]
                kernels.append(valid_img)
            imgs.append(kernels)
        x = torch.from_numpy(np.array(imgs))
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


def build_gabor():
    filters = []

    for theta in np.arange(0, np.pi, np.pi / 2):
        for k in range(5):
            kern = cv2.getGaborKernel((5, 5), 1.0, theta, np.pi / 2.0, 0.5, 0, ktype = cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters


def main():
    kernels = build_gabor()

    ex_network = NetworkGabor(kernels)
    loaded_net_state_dict = torch.load('ex_model.pt')
    ex_network.load_state_dict(loaded_net_state_dict)

    train_loader, test_loader = deepNetwork.read_and_print(deepNetwork.BATCH_SIZE_TRAIN, deepNetwork.BATCH_SIZE_TEST,
                                                           False)
    ex_network.eval()
    test_loss = 0
    correct = 0
    extracted_examples = []
    extracted_preds = []
    extracted_targets = []
    # disable gradient calculation is useful for inference, backward() will not be called in testing
    with torch.no_grad():
        c = 0
        for data, target in test_loader:
            output = ex_network(data)
            for i in range(deepNetwork.BATCH_SIZE_TEST):
                if c >= 10:
                    break
                extracted_examples.append(data[i])
                extracted_preds.append(output[i])
                extracted_targets.append(target[i])
                c += 1
            test_loss += F.cross_entropy(output, target, reduction = 'sum').item()
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)

    print('\nTest over entire test set: Avg.loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # run the model through the first 10 examples
    examples = torch.stack(extracted_examples)
    preds = torch.stack(extracted_preds)
    targets = torch.stack(extracted_targets)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        output = ex_network(examples)
        print("output: ", end = "")
        test_loss += F.cross_entropy(output, targets, reduction = 'sum').item()
        pred = output.data.max(1, keepdim = True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
    test_loss /= len(examples)

    print('\nTest over entire test set: Avg.loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(examples), 100. * correct / len(examples)))

    for i in range(len(examples)):
        print(f"\nExample {i + 1}: \nOutput values :", end = "")
        print(['%.2f' % t for t in preds[i]])
        print("Predicted label: %d" % [t for t in preds[i]].index(max(preds[i])))
        print("Correct label: %d" % targets[i].item())

    fig = plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(examples[i][0], cmap = 'gray', interpolation = 'none')
        plt.title("Prediction: %d" % ([t for t in preds[i]].index(max(preds[i]))))
        plt.xticks([])
        plt.yticks([])
    fig.show()


if __name__ == '__main__':
    main()
