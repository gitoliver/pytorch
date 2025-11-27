import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Lets see if I can adapt this to run on my GPU instead of CPUs
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# This is validation to check if model performance is improving
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # I guess they have separate datasets?
    download=True,
    transform=ToTensor()
)

# Batch size: Number of samples to put through network before updating parameters
train_dataloader = DataLoader(training_data, batch_size=64) 
test_dataloader = DataLoader(test_data, batch_size=64)

# Same NN as before, see earlier notes.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#model = NeuralNetwork()
model = NeuralNetwork().to(device) # GPU

# Hyperparameters
learning_rate = 1e-3 # Smells like steps size in gradient descent. Bigger gives unpredictability, smaller takes a lot of time.
batch_size = 64
epochs = 5 # Number of cycles/iterations over the entire dataset. Batch size is update frequency.

# Loss functions:
## Common loss functions include nn.MSELoss (Mean Square Error) for regression tasks, and nn.NLLLoss (Negative Log Likelihood) for classification. nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.
loss_fn = nn.CrossEntropyLoss()

# Optimizer: Stochastic Gradient Descent, there are others that are data/model specific.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)  # Move data to GPU
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  # Move data to GPU
            pred = model(X) # Pass batch X through the NN. Pred will be shaped: (batch_size, *output_dimensions)
            test_loss += loss_fn(pred, y).item() 
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # Count how many max probabilities in pred match the correct answer in y. The type(torch.float) is converting from True/False to 1.0/0.0. Item extracts a number from the tensor i.e. 2.0 becomes 2.

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
