import torch
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from neuralnet import MLP, Trainer, save_model

def main():
 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    layer_sizes    = [28*28, 512, 256, 128, 10]
    dropout        = 0.5
    batch_norm     = True
    batch_size     = 128
    epochs         = 30
    lr             = 1e-3
    optimizer      = 'adam'
    weight_decay   = 1e-4
    lr_decay       = 0.95       
    early_stopping = True
    patience       = 5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.EMNIST(
        root='./data',
        split='digits',
        train=True,
        download=True,
        transform=transform
    )
    test_set   = datasets.EMNIST(
        root='./data',
        split='digits',
        train=False,
        download=True,
        transform=transform
    )

    n_train = int(0.9 * len(full_train))
    n_val   = len(full_train) - n_train
    train_set, val_set = random_split(full_train, [n_train, n_val])

    model   = MLP(layer_sizes, dropout=dropout, batch_norm=batch_norm)
    trainer = Trainer(model, device=device)

    history = trainer.fit(
        train_dataset   = train_set,
        val_dataset     = val_set,
        batch_size      = batch_size,
        epochs          = epochs,
        lr              = lr,
        optimizer_name  = optimizer,
        weight_decay    = weight_decay,
        lr_decay        = lr_decay,
        early_stopping  = early_stopping,
        patience        = patience
    )

    save_path = "emnist_digits_model.pth"
    save_model(model, save_path)
    print(f"Model saved to '{save_path}'")

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'],   label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
