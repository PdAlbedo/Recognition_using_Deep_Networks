"""
Write out the ten digits [0-9] in your own handwriting on a piece of white paper (not too close together).
You will want to use thick (really thick) lines when writing the digits. I suggest using a marker or sharpie.
Writing them on a white board may also work. Take a picture of the digits, crop each digit to its own square image.
Test the network on new inputs
"""
__author__ = "Xichen Liu"

import os
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from matplotlib import pyplot as plt
import deepNetwork

# from PIL import Image
# import PIL.ImageOps


torch.manual_seed(888)


class HandWriteDateset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).float()
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def main():
    # convert the image to the correct size and format if the digits are the black on white
    # for i in range(10):
    #     image = Image.open('test_cases/handWrite/%d.png' % i)
    #     image = image.resize((28, 28))
    #     gray_image = image.convert('L')
    #     inverted_image = PIL.ImageOps.invert(gray_image)
    #     inverted_image.save('test_cases/handWrite/%d.png' % i)

    loaded_net = deepNetwork.MyNetwork()
    loaded_net_state_dict = torch.load('model.pt')
    loaded_net.load_state_dict(loaded_net_state_dict)

    hand_write_dateset = HandWriteDateset(annotations_file = 'test_cases/handWrite.csv',
                                          img_dir = 'test_cases/handWrite/')
    hand_write_dataloader = DataLoader(dataset = hand_write_dateset,
                                       batch_size = deepNetwork.BATCH_SIZE_TEST,
                                       shuffle = False,
                                       num_workers = 4)

    # set model to eval()
    loaded_net.eval()
    test_loss = 0
    correct = 0
    imgs = []
    preds = []
    # disable gradient calculation is useful for inference, backward() will not be called in testing
    with torch.no_grad():
        for data, target in hand_write_dataloader:
            output = loaded_net(data)
            test_loss += F.cross_entropy(output, target, reduction = 'sum').item()
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            imgs.append(data)
            preds.append(pred)

    # noinspection PyTypeChecker
    set_len = len(hand_write_dataloader.dataset)
    # batch_num = 0
    if (set_len / deepNetwork.BATCH_SIZE_TEST).is_integer():
        batch_num = set_len / deepNetwork.BATCH_SIZE_TEST
    else:
        batch_num = int(set_len / deepNetwork.BATCH_SIZE_TEST) + 1
    test_loss /= set_len

    print('\nTest over handwrite test set: Avg.loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, set_len, 100. * correct / set_len))

    fig = plt.figure()
    i = 0
    c = 0
    while i < batch_num:
        for j in range(deepNetwork.BATCH_SIZE_TEST):
            if c >= set_len:
                i = batch_num
                break
            plt.subplot(4, 3, c + 1)
            plt.tight_layout()
            plt.imshow(imgs[i][j][0], cmap = 'gray', interpolation = 'none')
            plt.title("Prediction: %d" % preds[i][j])
            plt.xticks([])
            plt.yticks([])
            c += 1
        i += 1
    fig.show()


if __name__ == '__main__':
    main()
