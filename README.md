# diffusion_for_fusion

## Installation instructions on Mac M3
1. Install pyenv with brew
    ```
    brew install pyenv
    ```
    Then follow the setup instructions of pyenv on the pyenv github page. 
2. Install python 3.10.1
    ```
    pyenv install 3.10.1
    ```
3. Make a directory to keep everything in,
    ```
    mkdir project
    cd project
    ```
4. Set the python and make a virtualenv,
    ```
    pyenv local 3.10.1
    python -m venv env
    source env/bin/activate
    ```
   The venv can be activated with `source env/bin/activate`.
5. Install some pip packages
    ```
    python -m pip install numpy 
    python -m pip install sklearn scipy tqdm pyparsing pandas celluloid matplotlib
    python -m pip install torch
    python -m pip install torchvision
    ```
    To use simsopt,
    ```
    brew install open-mpi
    python -m pip install cmake scikit-build ninja f90wrap
    python -m pip install mpi4py
    python -m pip install simsopt"[MPI]"
    ```
6. Export the python path so that import statements work properly. The following will overwrite your PYTHONPATH.
    ```
    export PYTHONPATH="/Users/mpadidar/code/ml/diffusion_for_fusion"
    ```
    To append to your PYTHONPATH, use
    ```
    export PYTHONPATH="${PYTHONPATH}:/Users/mpadidar/code/ml/diffusion_for_fusion"
    ```
