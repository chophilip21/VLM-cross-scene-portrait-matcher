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

Now we need to confirm that our code works first. You can run the application in debug mode directly via:

```bash
# look into config.ini 
source env/Scripts/activate
python main.py
```

# package

Pyinstaller is used for packaging.

```bash
chmod +x helpers/package.sh
make package
```