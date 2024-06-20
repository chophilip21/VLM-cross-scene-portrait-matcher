# Getting started with Pyinside6

```bash
conda create --name pyinside python=3.11
make build
make install
sudo apt-get install libxcb-cursor0
```

You can run the application directly via:

```bash
# look into config.ini 
python main.py
```

# Packaging the app

```bash
pip3 install PyInstaller

# build without spec file 
pyinstaller --name photolink --onefile main.py

# build against spec file
pyinstaller photolink.spec
```