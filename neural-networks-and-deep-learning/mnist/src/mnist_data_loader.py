import cPickle, gzip, numpy


def load_data():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_set, validation_set, test_set = cPickle.load(f)
    f.close()
    return training_set, validation_set, test_set


def load_data_customized():
    training_set, validation_set, test_set = load_data()
    training_x = [numpy.reshape(x, (784, 1)) for x in training_set[0]]
    training_y = [vectorize_y(y) for y in training_set[1]]
    training_data = zip(training_x, training_y)

    validation_x = [numpy.reshape(x, (784, 1)) for x in validation_set[0]]
    validation_data = zip(validation_x, validation_set[1])

    test_x = [numpy.reshape(x, (784, 1)) for x in test_set[0]]
    test_data = zip(test_x, test_set[1])

    return training_data, validation_data, test_data


def vectorize_y(i):
    y = numpy.zeros((10, 1))
    y[i] = 1
    return y
