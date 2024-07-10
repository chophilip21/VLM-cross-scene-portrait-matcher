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

In order to use `Faiss` for fast vector search and export it to Windows without error, we need to build it from source.

A requirement for this is to have [MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=windows&windows-install-type=online) library installed.



Now we need to confirm that our code works first. You can run the application in debug mode directly via:

```bash
# look into config.ini 
source env/Scripts/activate
python launch.py
```

# Packaging to executables

The code, dependencies, and assets all need to be converted into executables first. Assert it runs the same as debug mode.


```bash
chmod +x helpers/package.sh
make package
```

## Distributing to Windows Users (EXE file)

Now collected execution files need to be wrapped up for Windows users, so that they can download with an installer.

To distribute the `.exe` file to Windows Users in a standard way, you need [INNO setup tools](https://jrsoftware.org/isdl.php). Just download the installer and feed in your files based on Wizard prompt to generate `.iss` Pascal script that can be auto-compiled into a Windows installer by Inno setup tools. You do not have to code anything, just ensure that you are passing the `_internals` folder correctly, which contains all the dependencies. Everything will be saved to Output folder in the project root.  
