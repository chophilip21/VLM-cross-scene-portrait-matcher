# Getting started with Pyinside6

The way you set things up on Linux and Windows is slightly different.

```bash
# general setup
chmod +x helpers/setup.sh
make setup

# if "make" is not installed on Windows:
choco install make

# on linux, you may need to run this.
sudo apt-get install libxcb-cursor0
```

In order to use `Faiss` for fast vector search and export it to Windows without error, we need to build it from source. `Pip install faiss` will be very slow when you compile it using Nuitka. To install Faiss from source, a requirement for this is to have [MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=windows&windows-install-type=online) library installed. Read Nuitka [readme](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for more details. When MKL is installed properly, you should be able to run `make setup`.


If you get error during `make setup` for `faiss`, you can cd into faiss directory after git clone, and try running something like this manually:

```bash 
cmake -B build -DFAISS_ENABLE_GPU=OFF -DBLA_VENDOR=Intel10_64_dyn -DBLAS_LIBRARIES="C:\Program Files (x86)\Intel\oneAPI\mkl\2024.2\lib\mkl_intel_lp64.lib;C:\Program Files (x86)\Intel\oneAPI\mkl\2024.2\lib\mkl_sequential.lib;C:\Program Files (x86)\Intel\oneAPI\mkl\2024.2\lib\mkl_core.lib" -DLAPACK_LIBRARIES="C:\Program Files (x86)\Intel\oneAPI\mkl\2024.2\lib\mkl_intel_lp64.lib;C:\Program Files (x86)\Intel\oneAPI\mkl\2024.2\lib\mkl_sequential.lib;C:\Program Files (x86)\Intel\oneAPI\mkl\2024.2\lib\mkl_core.lib" -DPython_EXECUTABLE="C:\Users\choph\photomatcher\env\Scripts\python3.exe" .
```

Obviously you need to swap out the paths. And then

```bash
cmake --build build --config Release -j
(cd build/faiss/python && python setup.py install)
```

# Testing locally

Now we need to confirm that our code works first. You can run the application in debug mode directly via:

```bash
# look into config.ini 
source env/Scripts/activate
python launch.py
```

# Packaging to executables

This is the most tricky part, but I have managed to package the application using two tools:

1. Nuitka
2. Inno Setup (For windows)

The code, dependencies, and assets all need to be converted into executables first. Assert it runs the same as debug mode.

```bash
chmod +x helpers/package.sh
make package
```

## Distributing to Windows Users (Windows Installer file)

Now collected execution files need to be wrapped up for Windows users, so that they can download with an installer.

To distribute the `.exe` file to Windows Users in a standard way, you need [INNO setup tools](https://jrsoftware.org/isdl.php). Just download the installer and feed in your files based on Wizard prompt to generate `.iss` Pascal script that can be auto-compiled into a Windows installer by Inno setup tools. You do not have to code anything, just make sure you pass my `windows.iss` to Inno setup, and it will compile the installer for you. You can pass the output to the client. Getting code signing is very troublesome, and requires payment. Not worth it for this purpose.


