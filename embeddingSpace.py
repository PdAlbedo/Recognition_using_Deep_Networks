"""
Use the trained network as an embedding space for images of written symbols.
"""
__author__ = "Xichen Liu"

import os
import csv
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision.io import read_image
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import deepNetwork
import handWriteTest

torch.manual_seed(888)

CONV_FILTER_SIZE = [3, 5, 7]
DROPOUT_RATE = [0.3, 0.5, 0.7]
KNN_INPUT_FILENAME = 'test_cases_greek/handWriteGreek/phi03.png'


class Submodel(deepNetwork.MyNetwork):
    def __init__(self, conv_filter = 5, dropout_rate = 0.5):
        super().__init__(conv_filter, dropout_rate)

    # override the forward method
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
        # Stop at 50-node dense layer
        x = F.relu(self.fc1(self.flat1(x)))
        return x


def ssd(a, b):
    d = np.sum((a - b) ** 2)
    return d


def save_to_csv():
    """
    Save the intensity values and their labels into two csv files
    """
    with open('test_cases_greek/greek_intensity.csv', 'w', encoding = 'UTF8', newline = '') as f:
        writer = csv.writer(f)
        header = ['Filename', 'Intensity']
        writer.writerow(header)
        for filename in os.listdir('test_cases_greek/greek-1'):
            image = Image.open(os.path.join('test_cases_greek/greek-1', filename))
            writer.writerow([filename, np.array(image)])

    with open('test_cases_greek/greek_labels.csv', 'w', encoding = 'UTF8', newline = '') as f:
        writer = csv.writer(f)
        header = ['Filename', 'Label']
        writer.writerow(header)
        for filename in os.listdir('test_cases_greek/greek-1'):
            if 'alpha' in filename:
                writer.writerow([filename, 0])
            elif 'beta' in filename:
                writer.writerow([filename, 1])
            elif 'gamma' in filename:
                writer.writerow([filename, 2])
            elif 'eta' in filename:
                writer.writerow([filename, 3])
            elif 'phi' in filename:
                writer.writerow([filename, 4])

    greek_dataset = handWriteTest.HandWriteDateset(annotations_file = 'test_cases_greek/greek_labels.csv',
                                                   img_dir = 'test_cases_greek/greek-1')
    greek_dataloader = DataLoader(dataset = greek_dataset,
                                  batch_size = deepNetwork.BATCH_SIZE_TEST,
                                  shuffle = False,
                                  num_workers = 4)

    return greek_dataset, greek_dataloader


def build_embedding_space(greek_submodel, greek_dataloader):
    """
    Apply the truncated network to the greek symbols (read from the CSV file) to get a set of 27 50 element vectors

    :param greek_submodel:      submodel build from loaded network
    :param greek_dataloader:    dataloader of greek dataset
    :return  overall shape of outputs, and labels
    """
    greek_submodel.eval()

    results = []
    targets = []
    b = 0
    for data, target in greek_dataloader:
        output = greek_submodel(data)
        print("\nBatch %d:" % b)
        print("Input batch size: ", end = "")
        print(data.shape)
        print("Apply the submodel with 50-node dense layer to the data, "
              "we have the returned output with the shape of: ", end = "")
        print(output.shape)
        b += 1
        # make sure no matter what the batch size is, the results will always keep all outputs, in this case,
        # shape of 27 * 50; and targets will have the corresponding labels
        for i in range(len(output)):
            results.append(output[i])
            targets.append(target[i])
    print("\nShape of the output nodes from the submodel: ", end = "")
    print(torch.stack(results).shape)
    print("Number of the labels: ", end = "")
    print(torch.stack(targets).shape)

    return results, targets


def extract_3_greek(results, targets):
    """
    Extract 3 greek symbols

    :param results: outputs from submodel
    :param targets: labels corresponding to the results
    :return selected alpha, beta, and gamma
    """
    alpha = []
    beta = []
    gamma = []
    label = 99
    for i in range(len(results)):
        if targets[i].detach().numpy() != label and targets[i].detach().numpy() == 0:
            for j in results[i].detach().numpy():
                alpha.append(j)
            label = targets[i].detach().numpy()
        elif targets[i].detach().numpy() != label and targets[i].detach().numpy() == 1:
            for j in results[i].detach().numpy():
                beta.append(j)
            label = targets[i].detach().numpy()
        elif targets[i].detach().numpy() != label and targets[i].detach().numpy() == 2:
            for j in results[i].detach().numpy():
                gamma.append(j)
            label = targets[i].detach().numpy()

    return alpha, beta, gamma


def knn(k, results, targets, a):
    """
    Calculate the label of the given data by assigning it the label of k nn's label

    :param k:       k nearest neighbors
    :param results: outputs from submodel
    :param targets: labels corresponding to the results
    :param a:       given example
    """
    sum_alpha = []
    sum_beta = []
    sum_gamma = []
    sum_eta = []
    sum_phi = []
    for i in range(len(results)):
        d = ssd(np.array(a), results[i].detach().numpy())
        print("%.2f" % d, end = " ")
        if targets[i].detach().numpy() == 0:
            sum_alpha.append(d)
        elif targets[i].detach().numpy() == 1:
            sum_beta.append(d)
        elif targets[i].detach().numpy() == 2:
            sum_gamma.append(d)
        elif targets[i].detach().numpy() == 3:
            sum_eta.append(d)
        elif targets[i].detach().numpy() == 4:
            sum_phi.append(d)
    sum_alpha.sort()
    sum_beta.sort()
    sum_gamma.sort()
    sum_eta.sort()
    sum_phi.sort()
    print("\nTotal distance to top kth label alpha: %.2f" % sum(sum_alpha[: k]))
    print("Total distance to top kth label beta: %.2f" % sum(sum_beta[: k]))
    print("Total distance to top kth label gamma: %.2f" % sum(sum_gamma[: k]))
    print("Total distance to top kth label eta: %.2f" % sum(sum_eta[: k]))
    print("Total distance to top kth label phi: %.2f" % sum(sum_phi[: k]))

    pred = min(sum(sum_alpha[: k]), sum(sum_beta[: k]),
               sum(sum_gamma[: k]), sum(sum_eta[: k]),
               sum(sum_phi[: k]))

    if pred == sum(sum_alpha[: k]):
        return "alpha"
    elif pred == sum(sum_beta[: k]):
        return "beta"
    elif pred == sum(sum_gamma[: k]):
        return "gamma"
    elif pred == sum(sum_eta[: k]):
        return "eta"
    elif pred == sum(sum_phi[: k]):
        return "phi"
    else:
        return "unknown"


def calculate_avg(results, targets, a):
    """
    Calculate the average distance from a given example to all of other examples

    :param results: outputs from submodel
    :param targets: labels corresponding to the results
    :param a:       given example
    """
    avg_alpha = []
    avg_beta = []
    avg_gamma = []
    avg_eta = []
    avg_phi = []
    for i in range(len(results)):
        d = ssd(np.array(a), results[i].detach().numpy())
        print("%.2f" % d, end = " ")
        if d == 0:
            continue
        if targets[i].detach().numpy() == 0:
            avg_alpha.append(d)
        elif targets[i].detach().numpy() == 1:
            avg_beta.append(d)
        elif targets[i].detach().numpy() == 2:
            avg_gamma.append(d)
        elif targets[i].detach().numpy() == 3:
            avg_eta.append(d)
        elif targets[i].detach().numpy() == 4:
            avg_phi.append(d)

    print("\nAverage distance to label alpha: %.2f" % (sum(avg_alpha) / len(avg_alpha)))
    print("Average distance to label beta: %.2f" % (sum(avg_beta) / len(avg_beta)))
    print("Average distance to label gamma: %.2f" % (sum(avg_gamma) / len(avg_gamma)))
    print("Average distance to label eta: %.2f" % (sum(avg_eta) / len(avg_eta)))
    print("Average distance to label phi: %.2f" % (sum(avg_phi) / len(avg_phi)))

    pred = min((sum(avg_alpha) / len(avg_alpha)), (sum(avg_beta) / len(avg_beta)),
               (sum(avg_gamma) / len(avg_gamma)), (sum(avg_eta) / len(avg_eta)),
               (sum(avg_phi) / len(avg_phi)))

    if pred == (sum(avg_alpha) / len(avg_alpha)):
        return "alpha"
    elif pred == (sum(avg_beta) / len(avg_beta)):
        return "beta"
    elif pred == (sum(avg_gamma) / len(avg_gamma)):
        return "gamma"
    elif pred == (sum(avg_eta) / len(avg_eta)):
        return "eta"
    elif pred == (sum(avg_phi) / len(avg_phi)):
        return "phi"
    else:
        return "unknown"


def eval_hand_write_greek(greek_submodel, results, targets, is_knn, s, dr):
    hand_write_greeks = handWriteTest.HandWriteDateset(annotations_file = 'test_cases_greek/handWriteGreek.csv',
                                                       img_dir = 'test_cases_greek/handWriteGreek/')
    hand_write_greeks_dataloader = DataLoader(dataset = hand_write_greeks,
                                              batch_size = deepNetwork.BATCH_SIZE_TEST,
                                              shuffle = False,
                                              num_workers = 4)

    print("\nBuilding embedding space for hand write greeks")
    hand_write_results, hand_write_targets = build_embedding_space(greek_submodel, hand_write_greeks_dataloader)

    imgs = []
    for data, target in hand_write_greeks_dataloader:
        imgs.append(data)

    if (len(hand_write_results) / deepNetwork.BATCH_SIZE_TEST).is_integer():
        batch_num = len(hand_write_results) / deepNetwork.BATCH_SIZE_TEST
    else:
        batch_num = int(len(hand_write_results) / deepNetwork.BATCH_SIZE_TEST) + 1

    preds = []
    for i in range(len(hand_write_results)):
        print("\nDistance from selected image %d to others: " % i)
        # preds.append(calculate_avg(hand_write_results, hand_write_targets, hand_write_results[i].detach().numpy()))
        if is_knn:
            preds.append(knn(7, results, targets, hand_write_results[i].detach().numpy()))
        else:
            preds.append(calculate_avg(results, targets, hand_write_results[i].detach().numpy()))

    i = 0
    c = 0
    fig = plt.figure()
    while i < batch_num:
        for j in range(deepNetwork.BATCH_SIZE_TEST):
            if c >= len(hand_write_results):
                i = batch_num
                break
            plt.subplot(3, 5, c + 1)
            plt.tight_layout()
            plt.imshow(imgs[i][j][0], cmap = 'gray', interpolation = 'none')
            if is_knn:
                plt.title("s{}dr{}kPred:\n{}".format(s, dr, preds[c]))
            else:
                plt.title("s{}dr{}Pred:\n{}".format(s, dr, preds[c]))
            plt.xticks([])
            plt.yticks([])
            c += 1
        i += 1
    if is_knn:
        plt.savefig('results/Fsz_{}_Drt_p{}knn.png'.format(s, int(dr * 10)))
    else:
        plt.savefig('results/Fsz_{}_Drt_p{}.png'.format(s, int(dr * 10)))
    fig.show()


def pred_by_knn(filename, k, greek_submodel, results, targets):
    image = read_image(filename).float()
    image = image[None, :]
    output = greek_submodel(image)
    print('\nDistance from input image to greek database:')
    pred = knn(k, results, targets, output[0].detach().numpy())
    fig = plt.figure()
    plt.imshow(image[0][0], cmap = 'gray', interpolation = 'none')
    plt.title("Prediction: %s" % pred)
    plt.xticks([])
    plt.yticks([])
    fig.show()


def main():
    # convert the images into proper format
    # for filename in os.listdir('test_cases_greek/handWriteGreek'):
    #     image = Image.open(os.path.join('test_cases_greek/handWriteGreek', filename))
    #     image = image.resize((28, 28))
    #     image = image.convert('L')
    #     # image = ImageOps.invert(image)
    #     image.save(os.path.join('test_cases_greek/handWriteGreek', filename))
    #
    #     img = cv2.imread(os.path.join('test_cases_greek/handWriteGreek', filename))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     (row, col) = img.shape[0:2]
    #     for i in range(row):
    #         for j in range(col):
    #             if img[i, j] < 175:
    #                 img[i, j] = 0
    #             else:
    #                 img[i, j] = 255
    #     cv2.imwrite(os.path.join('test_cases_greek/handWriteGreek', filename), img)

    # load the model
    loaded_net = deepNetwork.MyNetwork()
    loaded_net_state_dict = torch.load('model.pt')
    loaded_net.load_state_dict(loaded_net_state_dict)
    loaded_net.eval()

    greek_dataset, greek_dataloader = save_to_csv()

    for i in CONV_FILTER_SIZE:
        for j in DROPOUT_RATE:
            print("#################################################################")
            print("For kernel size = {} and drop rate = {}\n".format(i, j))

            greek_submodel = Submodel(i, j)

            print("Building embedding space of greek-1")
            results, targets = build_embedding_space(greek_submodel, greek_dataloader)
            print("Embedding space of greek-1 built")

            alpha, beta, gamma = extract_3_greek(results, targets)

            print("\nDistance from selected alpha to others: ")
            calculate_avg(results, targets, alpha)
            print("\nDistance from selected beta to others: ")
            calculate_avg(results, targets, beta)
            print("\nDistance from selected gamma to others: ")
            calculate_avg(results, targets, gamma)
            print('\n')

            eval_hand_write_greek(greek_submodel, results, targets, False, i, j)
            eval_hand_write_greek(greek_submodel, results, targets, True, i, j)
            print("#################################################################\n")

    greek_submodel = Submodel()
    results, targets = build_embedding_space(greek_submodel, greek_dataloader)
    pred_by_knn(KNN_INPUT_FILENAME, 3, greek_submodel, results, targets)


if __name__ == '__main__':
    main()
