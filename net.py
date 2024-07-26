import numpy as np
import scipy.special as sp


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate
        self.wih = np.random.normal(
            0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes)
        )
        self.who = np.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes)
        )
        self.act_fn = lambda x: sp.expit(x)

    def train(self, inputs_list, targets_list):
        targets = np.array(targets_list, ndmin=2).T
        inputs, hidden_outputs, final_outputs = self.query(inputs_list)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot(
            (output_errors * final_outputs * (1 - final_outputs)),
            np.transpose(hidden_outputs),
        )
        self.wih += self.lr * np.dot(
            (hidden_errors * hidden_outputs * (1 - hidden_outputs)),
            np.transpose(inputs),
        )

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        # Calculate signals into and out of hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.act_fn(hidden_inputs)

        # Calculate signals into and out of output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.act_fn(final_inputs)

        return inputs, hidden_outputs, final_outputs
