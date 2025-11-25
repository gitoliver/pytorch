import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
deviceName = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"Using {deviceName} device")
device = "cuda"


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # Down to 1 dimension.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # Input layer of flattened image. Linear transformation using stored weights and biases.
            nn.ReLU(), # Activation function to use. -ve become 0, +ve stays the same. Breaks linearity when you have many neurons.
            nn.Linear(512, 512), # Layer 2.
            nn.ReLU(),
            nn.Linear(512, 10), # Output layer
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits 

model = NeuralNetwork().to(device) # This creates the NN with random weights and biases.
# Can either load weights or train it, here they just use it with the random values.
print(model)

X = torch.rand(1, 28, 28, device=device) # Random pixels
logits = model(X) # Puts image through model and gets logits.
pred_probab = nn.Softmax(dim=1)(logits) # Converts logits to 0-1 values for probabilty.
y_pred = pred_probab.argmax(1) # Finds to highest probability answer.
print(f"Predicted class: {y_pred}") # Prints it.


