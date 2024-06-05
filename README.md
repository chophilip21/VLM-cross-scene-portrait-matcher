# TogaMatcher

Create a webapp using Toga. For more info, refer to [this](https://docs.beeware.org/en/latest/tutorial/tutorial-0.html)

```bash
python3 -m venv beeware-venv
source beeware-venv/bin/activate
python -m pip install briefcase
```

You may need to update some packages required to run briefcase commands:

```bash
sudo apt-get update
sudo apt install libcairo2-dev pkg-config python3-dev
sudo apt install cmake
sudo apt install libgirepository1.0-dev
pip install pycairo
```

You can do some basic testing of the application by:

```bash
# run as dev debug mode.
cd photomatcher
briefcase dev
```

