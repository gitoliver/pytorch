# Neural Network Background info:
I watched the Neural Networks videos series from 3Blue1Brown.
https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

See the notes I made as I watched here [backgroundNotes.md](backgroundNotes.md)

# Venv and Tutorial Setup Notes:
For setup my GPU is too old for the latest version of pytorch, so I had to install an earlier version. 
See below in the virtual env setup or look in [requirements.txt](requirements.txt)

My GPU: NVIDIA GeForce GTX 1060 6GB device
This required altering the tutorial code to not use the "accelerator" class or functions, but the older "cuda" one.

## Python venv motivation
For tutorial 2, I got this error:
python3 -i tutorial2_DatasetsAndLoaders.py 
/home/oliver/.local/lib/python3.10/site-packages/matplotlib/projections/__init__.py:63: UserWarning: Unable to import Axes3D. This may be due to multiple versions of Matplotlib being installed (e.g. as a system package and as a pip package). As a result, the 3D projection is not available.
  warnings.warn("Unable to import Axes3D. This may be due to multiple versions of "

Essentially I should be setting up a venv for every python project I do, as the dependency issues get insane. 

## Python virtual environment setup
python3 -m venv pytorchTutorial
### activate:
source pytorchTutorial/bin/activate
pip install --upgrade pip
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install matplotlib
pip install "numpy<2" pandas pillow

### run the code:
python tutorial2_DatasetsAndLoaders.py

## Reminder, you activate and deactivate the venv like this:
source pytorchTutorial/bin/activate
deactivate


