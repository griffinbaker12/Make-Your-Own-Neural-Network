import numpy as np
import scipy.ndimage as snd

from constants import hidden_nodes, input_nodes, learning_rate, output_nodes
from net import NeuralNetwork


def train_and_eval(learning_rate, epochs=10):
    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    for _ in range(epochs):
        with open("mnist_dataset/mnist_train.csv", "r") as f:
            train_data = f.readlines()

        for record in train_data:
            all_values = record.split(",")
            inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            nn.train(inputs, targets)

            inputs_plus10_img = snd.rotate(
                inputs.reshape(28, 28), 10, cval=0.01, order=1, reshape=False
            )
            nn.train(inputs_plus10_img.reshape(784), targets)
            inputs_minus10_img = snd.rotate(
                inputs.reshape(28, 28), -10, cval=0.01, order=1, reshape=False
            )
            nn.train(inputs_minus10_img.reshape(784), targets)

    scorecard = []

    with open("mnist_dataset/mnist_test.csv", "r") as f:
        test_data = f.readlines()

    for record in test_data:
        all_values = record.split(",")
        correct_label = int(all_values[0])
        inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
        _, _, outputs = nn.query(inputs)
        label = np.argmax(outputs)
        scorecard.append(1 if label == correct_label else 0)

    return np.mean(scorecard)


print(f" Total accuracy: ", train_and_eval(learning_rate))

# learning_rates = np.logspace(-5, 0, 20)
#
# results = []
# for lr in learning_rates:
#     accuracy = train_and_eval(lr)
#     results.append(accuracy)
#
#
# # Find the best learning rate
# best_lr = learning_rates[np.argmax(results)]
# print(f"Best learning rate: {best_lr:.6f}")
#
# plt.figure(figsize=(10, 6))
# plt.semilogx(learning_rates, results, "bo-")
#
# # Customize x-axis
# plt.xlabel("Learning Rate")
# plt.xscale("log")
# plt.xlim(1e-5, 1)  # Adjust this range based on your actual learning_rates
#
# # Customize y-axis
# plt.ylabel("Performance")
# plt.ylim(0.8, 0.98)
# plt.yticks(np.arange(0.8, 0.981, 0.02))
#
# plt.title("Model Performance vs Learning Rate")
# plt.grid(True)
#
# # Add annotations for min and max performance
# min_perf = min(results)
# max_perf = max(results)
# plt.annotate(
#     f"Min: {min_perf:.4f}",
#     xy=(learning_rates[np.argmin(results)], min_perf),
#     xytext=(0.1, 0.1),
#     textcoords="axes fraction",
#     arrowprops=dict(facecolor="black", shrink=0.05),
# )
# plt.annotate(
#     f"Max: {max_perf:.4f}",
#     xy=(learning_rates[np.argmax(results)], max_perf),
#     xytext=(0.1, 0.9),
#     textcoords="axes fraction",
#     arrowprops=dict(facecolor="black", shrink=0.05),
# )
#
# plt.tight_layout()
# plt.show()
