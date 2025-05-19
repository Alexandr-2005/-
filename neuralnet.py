import random
import numpy as np
import pickle

# Активизация и её производная

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        n = len(training_data)
        
        if test_data is not None:
            test_data = list(test_data)
            n_test = len(test_data)
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # После каждой эпохи выводим статистику
            if test_data is not None:
                accuracy = self.evaluate(test_data)
                print(f"Epoch {j+1}/{epochs}: {accuracy} / {n_test} correct ({accuracy/n_test:.4%})")
            else:
                print(f"Epoch {j+1}/{epochs} complete")

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b - (eta / len(mini_batch)) * nb
                        for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # обратное распространение ошибки
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        correct = 0
        for x, y in test_data:
            pred = np.argmax(self.feedforward(x))
            true = np.argmax(y) if hasattr(y, 'ndim') and y.ndim > 0 else int(y)
            if pred == true:
                correct += 1
        return correct

    def cost_derivative(self, output_activations, y):
        return output_activations - y


# Сохраняем и загружаем модель при необходимости

def save_model(net, path):
    with open(path, 'wb') as f:
        pickle.dump(net, f)

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)