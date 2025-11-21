# Background info:
I watched the Neural Networks videos series from 3Blue1Brown.
https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

See [backgroundNotes.md](backgroundNotes.md)

# Setup Notes:
For setup my GPU is too old for the latest version of pytorch, so I had to install an earlier version. See below or requirements.txt
Using NVIDIA GeForce GTX 1060 6GB device

For tutorial 2, I got this error:
python3 -i tutorial2_DatasetsAndLoaders.py 
/home/oliver/.local/lib/python3.10/site-packages/matplotlib/projections/__init__.py:63: UserWarning: Unable to import Axes3D. This may be due to multiple versions of Matplotlib being installed (e.g. as a system package and as a pip package). As a result, the 3D projection is not available.
  warnings.warn("Unable to import Axes3D. This may be due to multiple versions of "

# Python virtual environment setup
python3 -m venv pytorchTutorial
activate:
source pytorchTutorial/bin/activate
pip install --upgrade pip
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install matplotlib
pip install "numpy<2" pandas pillow

python tutorial2_DatasetsAndLoaders.py

# Reminder, you activate and deactivate the venv like this:
source pytorchTutorial/bin/activate
deactivate


