# Getting started with Pyinside6

The way you set things up on Linux and Windows is slightly different. Windows make life hard for devs.

First, install Chocolatey. This acts like homebrew for Mac. Open command prompt as admin.

```bash
@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "[System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
```

You may have to run:

```bash
nano ~/.bash_profile
export PATH=$PATH:/c/ProgramData/chocolatey/bin
```

If you don't have it already, you also need the [C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) for some libraries, as well as [Cmake](https://cmake.org/download/).


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


# TODO List

✅ Replace `faiss` that was difficult to build with `nmslib-metabrainz`
✅ Implement function to read images safely using metadata
❌ Create the yoloworld feature. work on the pipeline. Call in `DP2 match` for now. Download the weights efficienly from dropbox. 
❌ Think about ways to obfuscate credentials. 
❌ Replace yunet with scrfd for better performance. 

