# What is this?
An MNIST classifier!

The final architecture ended up being:
- 284 input layers
- 200 hidden layers
- 10 output layers

The net was trained over 7 epochs with a learning rate of 0.297635 (optimal lr holding the epoch and hidden layer counts constant).

The final accuracy was 98.44%.

# Updates
Keeping the architecture constant, I was able to boost the performance to over 99% (99.1%) by training the net:
- Over 10 epochs
- With a learning rate of 0.088587 (found to be optimal given the architecture and epoch count)
