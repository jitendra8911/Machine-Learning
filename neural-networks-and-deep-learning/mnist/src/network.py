import numpy as np
from random import shuffle


class Network(object):

    def __init__(self, sizes):
        self.num_of_layers = len(sizes)
        self.weights = [np.random.randn(n, m) for n, m in zip(sizes[1:], sizes[:-1])]
        self.biases = [np.random.randn(n, 1) for n in sizes[1:]]
        self.sizes = sizes

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            # print('w shape', w.shape)
            # print('activation shape', activation.shape)
            # print('b shape', b.shape)
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_of_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)

    def SGD(self, samples, epochs, mini_batch_size, learning_rate, test_data=None):

        n_test = len(test_data)
        n = len(samples)
        for i in xrange(epochs):
            shuffle(samples)
            mini_batches = [
                samples[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_weights_biases(mini_batch, learning_rate)
            if test_data:
                print 'epoch {0}: {1}/{2}'.format(i, self.evaluate(test_data), n_test)
            else:
                print('epoch ', i)

    def update_weights_biases(self, mini_batch, learning_rate):

        sum_gradient_weights = [np.zeros(w.shape) for w in self.weights]
        sum_gradient_biases = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            gradient_biases, gradient_weights = self.backprop(x, y)
            sum_gradient_weights = [sum_gradient_w + gradient_w for sum_gradient_w, gradient_w in
                                    zip(sum_gradient_weights, gradient_weights)]
            sum_gradient_biases = [sum_gradient_b + gradient_b for sum_gradient_b, gradient_b in
                                   zip(sum_gradient_biases, gradient_biases)]

        self.weights = [w - (learning_rate / len(mini_batch)) * sum_gradient_w for w, sum_gradient_w in
                        zip(self.weights, sum_gradient_weights)]
        self.biases = [b - (learning_rate / len(mini_batch)) * sum_gradient_b for b, sum_gradient_b in
                       zip(self.biases, sum_gradient_biases)]

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
