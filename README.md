# TogaMatcher, getting started

Create a webapp using Toga. For more info, refer to [this](https://docs.beeware.org/en/latest/tutorial/tutorial-0.html). Seems the recommended version of python is 3.10. [Install it first if you need to](https://gist.github.com/rutcreate/c0041e842f858ceb455b748809763ddb).

```bash
python3.10 -m venv beeware-venv
source beeware-venv/bin/activate
pip install -r requirements
```

You may need to update some packages required to run briefcase commands:

```bash
sudo apt-get update
sudo apt install libcairo2-dev pkg-config python3-dev
sudo apt install cmake
sudo apt install libgirepository1.0-dev
```

You can do some basic testing of the application in the developer mode by:

```bash
# run as dev debug mode.
cd photomatcher
briefcase dev
```

# Shipping the application

When ready to ship the application, 

```bash
cd photomatcher
briefcase create
```

