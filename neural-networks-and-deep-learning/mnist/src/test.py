from network import Network
import numpy as np
import mnist_data_loader



training_set, validation_set, test_set = mnist_data_loader.load_data_customized()

net = Network([784, 30, 10])
#print('self.weights shape is ', net.weights[1].shape)


net.SGD(training_set, 30, 10, 3.0, test_set)





