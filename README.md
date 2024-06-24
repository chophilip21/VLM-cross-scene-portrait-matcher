# Getting started with Pyinside6

The way you set things up on Linux and Windows is slightly different.

```bash
# if using conda
conda create --name pyinside python=3.11

# if just venv
python -m venv env
source env/bin/activate # linux
source env/Scripts/activate # Windows. Ensure you have Choco package installer.

# if make is not installed on Windows:
choco install make

# then run
make build
make install

# on linux, you may need to run this.
sudo apt-get install libxcb-cursor0
```

Now we need to confirm that our code works first. You can run the application directly via:

```bash
# look into config.ini 
python main.py
```

# Packaging the app

The application works internally, so we need to package it using Pyinstaller.

```bash
pip3 install PyInstaller

# build without spec file 
pyinstaller --name photolink --log-level=DEBUG main.py

# build against spec file
pyinstaller photolink.spec
```