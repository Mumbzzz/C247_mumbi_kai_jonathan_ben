# My own readme for things i'll forget

He Kai Lim
UCLA Winter 2026



## Setting Up Environments etc.
- Use POWERSHELL. (there are shenanigans with WSL)
- Create the venv `python -m venv .venvProject`
- Activate with `.\.venvProject\Scripts\activate`
- Install correct pytorch:
  - `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
  - My computer takes cude version <= 12.9 so we select 12.8 from that matrix.>
  - Verson selected from matrix in https://pytorch.org/get-started/locally/
- Install packages with `pip install -r .\requirements.txt`
-  Connect with the kernel in ipynb.
-  Voila.
-  


## Anaconda

- open Anaconda Prompt (command prompt with (base))
  - `conda activate base`
- `conda create --name emg_project python=3.10`
- `conda activate emg_project`
- 

## [DEPRECATED] Setting Up Environments DEPRECATED - Does not work for the ipynb stuff.
- VSCode command pallete is ctrl + shift + p
- Create venv: `python3 -m venv .venvHW1`
- Activate venv: `source HW1/venvHW1/bin/activate`
- Install ipykernel: `pip install ipykernel`
- Install jupyter: `pip install jupyter`
- Install the required packages: `pip install -r HW1/installed-packages.txt`

## WSL things
- `sudo apt update && sudo apt upgrade`
- 