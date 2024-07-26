import numpy as np

from deep_net import DeepNeuralNetwork


def train_and_eval(learning_rate, epochs=30):
    layer_sizes = [784, 300, 200, 100, 10]
    nn = DeepNeuralNetwork(layer_sizes, learning_rate)

    for _ in range(epochs):
        with open("mnist_dataset/mnist_train.csv", "r") as f:
            train_data = f.readlines()

        for record in train_data:
            all_values = record.split(",")
            inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
            targets = np.zeros(10) + 0.01
            targets[int(all_values[0])] = 0.99
            nn.train(inputs, targets)

    scorecard = []

    with open("mnist_dataset/mnist_test.csv", "r") as f:
        test_data = f.readlines()

    for record in test_data:
        all_values = record.split(",")
        correct_label = int(all_values[0])
        inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
        outputs = nn.query(inputs)
        label = np.argmax(outputs)
        scorecard.append(1 if label == correct_label else 0)

    nn.save_model("deep_model.json")

    return np.mean(scorecard)


# learning_rate = 0.000010
# print(f" Total accuracy: ", train_and_eval(learning_rate))

learning_rates = np.logspace(-3, 0, 10)
results = []
for lr in learning_rates:
    accuracy = train_and_eval(lr)
    results.append(accuracy)


# Find the best learning rate
best_lr = learning_rates[np.argmax(results)]
print(f"Best learning rate: {best_lr:.6f}")
print(f"Accuracy: {results[np.argmax(results)]}")
