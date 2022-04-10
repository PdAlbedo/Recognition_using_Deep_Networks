"""
Read the network and run the model on the first 10 examples in the test set.
"""
__author__ = "Xichen Liu"

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import deepNetwork

torch.manual_seed(888)


def main():
    loaded_net = deepNetwork.MyNetwork()
    # loaded_optimizer = optim.SGD(loaded_net.parameters(), lr = learning_rate, momentum = momentum)

    loaded_net_state_dict = torch.load('model.pt')
    loaded_net.load_state_dict(loaded_net_state_dict)
    # loaded_optimizer_state_dict = torch.load('optimizer.pt')
    # loaded_optimizer.load_state_dict(loaded_optimizer_state_dict)

    train_loader, test_loader = deepNetwork.read_and_print(deepNetwork.BATCH_SIZE_TRAIN, deepNetwork.BATCH_SIZE_TEST,
                                                           False)

    # check if the model read from file is correct
    # set model to eval()
    loaded_net.eval()
    test_loss = 0
    correct = 0
    extracted_examples = []
    extracted_preds = []
    extracted_targets = []
    # disable gradient calculation is useful for inference, backward() will not be called in testing
    with torch.no_grad():
        c = 0
        for data, target in test_loader:
            output = loaded_net(data)
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
        output = loaded_net(examples)
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


if __name__ == "__main__":
    main()
