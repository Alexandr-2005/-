from neuralnet import Network, save_model
import mnist_loader
import matplotlib.pyplot as plt
import numpy as np


def main():
    print("1) Загружаем EMNIST (если нужно, скачиваем)...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print("2) Данные загружены — начинаем обучение.")
    
    sizes = [784, 256, 128, 64, 10]
    epochs = 5
    mini_batch_size = 32
    eta = 0.5

    # Загружаем данные (EMNIST Digits через обновлённый loader)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    validation_data = list(validation_data)
    test_data = list(test_data)

    # Инициализируем и тренируем сеть
    net = Network(sizes)
    print("Starting training on EMNIST Digits...")
    net.SGD(training_data, epochs, mini_batch_size, eta, test_data=test_data)
    print("Training complete.")

    # Отобразим несколько примеров предсказаний
    samples = test_data[:5]
    images, titles = [], []
    for x, y_true in samples:
        y_pred = net.feedforward(x).argmax()
        images.append(x.reshape(28, 28))
        titles.append(f"true={y_true} pred={y_pred}")

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    save_model(net, "trained_network.pkl")
    print("Model saved to 'trained_network.pkl'")

if __name__ == '__main__':
    main()
