#Notes:
For setup my GPU is too old for the latest version of pytorch, so I had to install an earlier version
Using NVIDIA GeForce GTX 1060 6GB device


For tutorial 2, I got this error:
python3 -i tutorial2_DatasetsAndLoaders.py 
/home/oliver/.local/lib/python3.10/site-packages/matplotlib/projections/__init__.py:63: UserWarning: Unable to import Axes3D. This may be due to multiple versions of Matplotlib being installed (e.g. as a system package and as a pip package). As a result, the 3D projection is not available.
  warnings.warn("Unable to import Axes3D. This may be due to multiple versions of "

So I should setup a virtual python environment for this.
setup: 
python3 -m venv pytorchTutorial
activate:
source pytorchTutorial/bin/activate
pip install --upgrade pip
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install matplotlib
pip install "numpy<2" pandas pillow

python tutorial2_DatasetsAndLoaders.py

deactivate


Working my way through this one and stopping:
https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
