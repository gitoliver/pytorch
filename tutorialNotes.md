
### Tutorial 4
That escalated quickly! This is a feedforward (info going in one direction only, no loops or backward connections) NN. The architecture is 3 layers.  
Input layer of 28x28flattened pixels resulting 784 neurons and then 512 out features. Linear(in_features=784, out_features=512, bias=True).  
Layer 2 512 neurons nn.Linear(512,512) 
Layer 3 Linear(in_features=512, out_features=10, bias=True)

We initialize the nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values.

The ReLU in between 
