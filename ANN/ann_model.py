# load libraries
import numpy as np
np.random.seed(42)

# create neuron class
class neuron(object):
    def __init__(self, num_inputs, activation):
        super().__init__()
        self.w = np.random.rand(num_inputs)
        self.b = np.random.rand(1)
        self.activation_function = activation

    def output(self, x):
        z = np.dot(x, self.w)+self.b
        return self.activation_function(z)

# input size of neuron 4
input_size = 4

# assign activation function
# out there lots of functions 
# for this taks I pick sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# define the model
model = neuron(num_inputs = input_size, activation=sigmoid)

# create random data 
x_data = np.random.rand(input_size).reshape(1, input_size)

y_output = model.output(x_data)
print(y_output)

# really good our model working without error
# right this is very simple model
# but now I am hungry, so I have to eat something
# later in this series I will make real world problems solwer with ANN
# raufodilov