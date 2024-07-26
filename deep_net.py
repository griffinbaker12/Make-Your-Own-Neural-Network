import json

import numpy as np
import scipy.special as sp


class DeepNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate, l2_reg=0.0001):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.lr = learning_rate
        self.l2_reg = l2_reg

        self.weights = [
            np.random.normal(
                0.0, pow(layer_sizes[i], -0.5), (layer_sizes[i + 1], layer_sizes[i])
            )
            for i in range(self.num_layers - 1)
        ]
        self.biases = [np.zeros((size, 1)) for size in layer_sizes[1:]]

        self.act_fn = lambda x: sp.expit(x)

    def train(self, inputs, targets):
        # Forward pass
        activations = self.forward_pass(inputs)

        # Backward pass
        delta = activations[-1] - targets.reshape(activations[-1].shape)
        for l in reversed(range(self.num_layers - 1)):
            # Calculate gradient
            grad = np.dot(delta, activations[l].T)

            # Update weights with L2 regularization
            self.weights[l] -= self.lr * (grad + self.l2_reg * self.weights[l])
            self.biases[l] -= self.lr * np.sum(delta, axis=1, keepdims=True)

            # Calculate delta for next layer
            if l > 0:
                delta = (
                    np.dot(self.weights[l].T, delta)
                    * activations[l]
                    * (1 - activations[l])
                )

    def forward_pass(self, inputs):
        activation = np.array(inputs, ndmin=2).T
        activations = [activation]

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            activation = self.act_fn(z)
            activations.append(activation)

        return activations

    def query(self, inputs):
        return self.forward_pass(inputs)[-1]

    def save_model(self, filename):
        model_data = {
            "layer_sizes": self.layer_sizes,
            "lr": self.lr,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        with open(filename, "w") as f:
            json.dump(model_data, f)

    @classmethod
    def load_model(cls, filename):
        with open(filename, "r") as f:
            model_data = json.load(f)

        nn = cls(model_data["layer_sizes"], model_data["lr"])
        nn.weights = [np.array(w) for w in model_data["weights"]]
        nn.biases = [np.array(b) for b in model_data["biases"]]
        return nn
