import torch
import numpy as np

def check_cuda():
    print(torch.version.cuda)
    cuda_is_ok = torch.cuda.is_available()
    print(f"CUDA Enabled: {cuda_is_ok}\n******\n")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    print("Note: You installed an older version of pytorch to get it to work with this\npip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117\n")

#Check cuda is cool
check_cuda()

#From data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

#From NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#From other tensor:
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

#Using shape. Shape is a tuple of tensor dimensions and determines tensor dimensionality.

shape = (2,3,4)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


# Tensor Attributes
# Tensor attributes describe their shape, datatype, and the device on which they are stored.
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the current accelerator if available
if torch.cuda.is_available():
    print("We ballin on the GPU now.\n")
    tensor = tensor.to("cuda")
print(f"Device tensor is now stored on: {tensor.device}")

#In-place operations
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
