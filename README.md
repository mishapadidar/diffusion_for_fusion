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
6. Install some stuff for working with VMEC,
    ```
    brew install gcc (this comes with gfortran)
    brew install scalapack
    brew install netcdf-fortran
    brew install openblas
    ```
    Also install f90wrap dependency meson-python:
    ```
    python -m pip install meson-python
    ```
7. clone VMEC2000
    ```
    git clone git@github.com:hiddenSymmetries/VMEC2000.git
    cd VMEC2000
    ```
8. Modify the `cmake_config_file.json` to be,
    ```
    {
        "cmake_args": [
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang",
            "-DCMAKE_Fortran_COMPILER=mpif90",
            "-DNETCDF_INC_PATH=/opt/homebrew/opt/netcdf-fortran/include",
            "-DNETCDF_LIB_PATH=/opt/homebrew/opt/netcdf-fortran/lib",
            "-DNETCDF_I=/opt/local/include/netcdf.inc"]
    }
    ```
    Then install VMEC
    ```
    python -m pip install . -v --no-cache-dir
    ```
    If the install fails, delete the `_skbuild` directory. Install whatever you need or adjust the `cmake_config_file.json` and try again. Sometimes, the pip install fails for frustrating reasons, in which case, try to build from source
    ```
    python setup.py build_ext
    python setup.py install
    ```
9. Export the python path so that import statements work properly.
    ```
    export PYTHONPATH="/Users/mpadidar/code/ml/diffusion_for_fusion"
    ```
