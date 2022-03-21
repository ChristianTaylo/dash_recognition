
#this code takes 4x5 inputs as training data, with a 5x1 training output, to train a program which can give a binary output for any input (of 1s and 0s)
#I've currently set it up so that it 'looks' for the presence of a dash i.e. any combination of two 1's next to eachother
#and returns a '1' if it detects a dash or '0' if no dash is detected. This code demonstrates the importance of training data
#as removing the last training sample causes the code to not correctly identify that sample as an input

import numpy as np

class NeuralNetwork():
    
    def __init__(self): #_init_ is equivalent to a 'constructor' in python. It is used to assign initial values to members of the class. "def" is used to mark the start of a function in python
        # seeding for random number generation
        np.random.seed(1)
        
        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((4, 1)) - 1
        self.bias = 2 * np.random.random() - 1

    def sigmoid(self, x):
        #applying the sigmoid function. this is the function which converts x into a number between 1 and 0 ie normalising the inputs. see maths notes
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        
        #training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            #siphon the training data via  the neuron
            output = self.think(training_inputs)

            #computing error rate for back-propagation
            error = training_outputs - output
            
            #performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))#i.e. this multiplies and sums each of the first input values by the error, followed by each of the second values etc.
            bias_adjustments = np.sum(error * self.sigmoid_derivative(output))#implemented this bit myself however it works. With weight adjustments, the use of dot product sums all the errors. As this doesn't use dot product, I sum manually
            #think of the above as kind've taking an average of errors. some will be negative, some positive, and by summing we descend on the optimal value
            self.bias += bias_adjustments
            self.synaptic_weights += adjustments 

    def think(self, inputs):
        #passing the inputs via the neuron to get output   
        #converting values to floats
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights) + self.bias) #dot product giving a 4x1 output. i.e. (sum(x1 *w1 +y1 *w2 + z1*w3) = output1, sum(x2 *w1 +y2 *w2 + z2*w3) = output 2 and so on)
        return output


if __name__ == "__main__":

    #initializing the neuron class
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights) #using the synaptic weights assigned in our NeuralNetwork class
    print("and Biases: ")
    print(neural_network.bias)
    #training data consisting of 4 examples--3 input values and 1 output
    training_inputs = np.array([[0,0,1,0], # = 0
                                [1,1,1,1], # = 1
                                [1,0,1,0], # = 0
                                [0,1,1,0], # = 1
                                [0,0,1,1]]) # = 1 accoridng to our output (below)

    training_outputs = np.array([[0,1,0,1,1]]).T # output

    #training taking place
    neural_network.train(training_inputs, training_outputs, 15000) #do the maths to adjust the wieghts of the neural network

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)
    print("and bias: ")
    print(neural_network.bias)

    user_input_one = str(input("User Input One: "))
    user_input_two = str(input("User Input Two: "))
    user_input_three = str(input("User Input Three: "))
    user_input_four = str(input("User Input Four: "))
    
    print("Considering New Situation: ", user_input_one, user_input_two, user_input_three, user_input_four)
    print("New Output data: ")
    print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three, user_input_four]))) #put the user input data through the algorithm to see if it correctly guesses
    print("Wow, we did it! Go again?")
    cont_test = str(input("Y/N "))

    while cont_test == 'Y': # added this bit so I could try other combinations if I wanted
        user_input_one = str(input("User Input One: "))
        user_input_two = str(input("User Input Two: "))
        user_input_three = str(input("User Input Three: "))
        user_input_four = str(input("User Input Four: "))

        print("Considering New Situation: ", user_input_one, user_input_two, user_input_three, user_input_four)
        print("New Output data: ")
        print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three, user_input_four])))
        cont_test = str(input("Y/N "))

    #code source https://www.kdnuggets.com/2018/10/simple-neural-network-python.html