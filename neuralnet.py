import numpy as np
import pickle

# Улучшенная реализация нейронной сети с использованием Xavier инициализации,
# ReLU в скрытых слоях, Softmax на выходе, кросс-энтропийной функции потерь и импульсом (momentum).

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

class ImprovedNetwork:
    def __init__(self, sizes, eta=0.01, alpha=0.9):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.eta = eta  # learning rate
        self.alpha = alpha  # momentum factor
        # Xavier initialization for weights and zero biases
        self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(2. / x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        self.vb = [np.zeros(b.shape) for b in self.biases]
        self.vw = [np.zeros(w.shape) for w in self.weights]

    def feedforward(self, a):

        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = w.dot(a) + b
            if i < self.num_layers - 2:
                a = relu(z)
            else:
                a = softmax(z)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta=None, test_data=None,
            decay=0.99, early_stopping=False, tol=5):

        if eta is not None:
            self.eta = eta
        n = len(training_data)
        best_acc = 0
        no_improve = 0

        for epoch in range(1, epochs + 1):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for batch in mini_batches:
                self.update_mini_batch(batch)

            self.eta *= decay

            if test_data:
                acc = self.evaluate(test_data)
                print(f"Epoch {epoch}/{epochs} — Accuracy: {acc}/{len(test_data)} ({acc/len(test_data):.2%}) — LR: {self.eta:.5f}")
                if early_stopping:
                    if acc > best_acc:
                        best_acc = acc
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= tol:
                            print("Early stopping triggered.")
                            return
            else:
                print(f"Epoch {epoch}/{epochs} complete. LR: {self.eta:.5f}")

    def update_mini_batch(self, mini_batch):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            db, dw = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, db)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, dw)]
        m = len(mini_batch)
        self.vw = [self.alpha * vw - (self.eta/m) * nw for vw, nw in zip(self.vw, nabla_w)]
        self.vb = [self.alpha * vb - (self.eta/m) * nb for vb, nb in zip(self.vb, nabla_b)]
        self.weights = [w + vw for w, vw in zip(self.weights, self.vw)]
        self.biases  = [b + vb for b, vb in zip(self.biases, self.vb)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []

        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = w.dot(activation) + b
            zs.append(z)
            activation = relu(z) if i < self.num_layers - 2 else softmax(z)
            activations.append(activation)

        delta = activations[-1] - y
        nabla_b[-1] = delta
        nabla_w[-1] = delta.dot(activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = self.weights[-l+1].T.dot(delta) * relu_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = delta.dot(activations[-l-1].T)

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        correct = 0
        for x, y in test_data:
            pred = np.argmax(self.feedforward(x))
            true = np.argmax(y) if hasattr(y, 'ndim') else int(y)
            if pred == true:
                correct += 1
        return correct

def save_model(net, path):
    with open(path, 'wb') as f:
        pickle.dump(net, f)

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
