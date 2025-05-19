import numpy as np
from torchvision import datasets, transforms

def load_emnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(784, 1))
    ])

    train_set = datasets.EMNIST(root='data',
                                split='digits',
                                train=True,
                                download=True,
                                transform=transform)
    test_set  = datasets.EMNIST(root='data',
                                split='digits',
                                train=False,
                                download=True,
                                transform=transform)

    training_data = [
        (img.numpy(), vectorized_result(label))
        for img, label in train_set
    ]
    test_data = [
        (img.numpy(), label)
        for img, label in test_set
    ]
    return training_data, None, test_data

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data_wrapper():
    tr, _, te = load_emnist_data()
    return (tr, [], te)
