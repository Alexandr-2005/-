from neuralnet import ImprovedNetwork, save_model
import mnist_loader
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("1) Загружаем EMNIST...")
    train_data, val_data, test_data = mnist_loader.load_data_wrapper()
    print("2) Нормализуем входы...")
    
    training_data = [(x / 255.0, y) for x, y in train_data]
    validation_data = [(x / 255.0, y) for x, y in val_data]
    test_data = [(x / 255.0, y) for x, y in test_data]

    sizes = [784, 128, 64, 10]
    epochs = 10
    mini_batch_size = 64
    eta = 0.01
    momentum = 0.9
    decay = 0.98
    early_stopping = True
    tol = 3

    print(f"3) Инициализируем ImprovedNetwork: слои={sizes}, lr={eta}, momentum={momentum}")
    net = ImprovedNetwork(sizes=sizes, eta=eta, alpha=momentum)

    print("4) Начинаем обучение...")
    net.SGD(
        training_data,
        epochs=epochs,
        mini_batch_size=mini_batch_size,
        eta=eta,
        test_data=test_data,
        decay=decay,
        early_stopping=early_stopping,
        tol=tol
    )
    print("Обучение завершено.")

    samples = test_data[:5]
    fig, axes = plt.subplots(1, len(samples), figsize=(10, 2))
    for ax, (x, y_true) in zip(axes, samples):
        pred_vec = net.feedforward(x)
        y_pred = np.argmax(pred_vec)
        if hasattr(y_true, 'ndim') and y_true.ndim > 0:
            true = np.argmax(y_true)
        else:
            true = int(y_true)
        img = x.reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"true={true} pred={y_pred}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    save_model(net, "trained_network.pkl")
    print("Model saved to 'trained_network.pkl'")

if __name__ == '__main__':
    main()
