Video 1: What is a Neural Network?
Take a 28x28 b&w image and make each point a neuron in your first layer. It's activation is how much white there is for that pixel.
Neuron activation is measured from 0.0 to 1.0.
Create a tensor of each of these 784 neurons.
Last layer has 10 neurons, each representing the 0-9 digits. How much that's activiated is the result.
Hidden layers in between are the magic. The number of layers and amount of neurons is open for experiment.

Activation in one layer determine activation in the next layer. In a very simple example, one would hope that the second to last layer corresponds to the components of the input image. So a loop on top for a 9 and an 8, then a line on bottom for a 9, which when combined activate the 9 in the last layer.

A hope might be that the second layer is activating for a lot of small components that make up the input image, and all these activations come together to activate the loop on top and line below in the third layer, which then activates 9 in the final layer.

It's all controlled by connections between the layers which have weights. So each node in the second layer is connected to each node in the first layer. Each connection has a weight, and how much activation a given node in the second layer gets is the sum of all activations in the first layer times the weights in the connections to this second layer node. 
Negative weight values can correspond to where you want darkness in the image, and postive for where you want brightness and you can then pick up on an edge in a particular part of the image.

You want the resulting value of this weighted sum to be between 0 and 1, so you use a function called a Sigmoid. You can also include a bias for the neuron to be inactive, so you can add or subject a number called a "bias". i.e. how high the weighted sum needs to be before you get activation of the neuron.

Remember! There are 784 neurons in the first layer, 16 in the second so 764 x 16 connections.

Learning is finding the right weights and balances. 

Math:
Organize all the activations in one layer into a vector, and all of the weights as a matrix. Each row is the set of weights between all the neurons in the first and one neuron in the second. So you're just doing matrix vector multiplication and it's linear algebra all the way down..
