
# Packaging to executables

This is the most tricky part, but I have managed to package the application using two tools:

1. Nuitka
2. Inno Setup (For windows)

The code, dependencies, and assets all need to be converted into executables first. Assert it runs the same as debug mode.

```bash
chmod +x helpers/package.sh
make nuitka
```

## Distributing to Windows Users (Windows Installer file)

Now collected execution files need to be wrapped up for Windows users, so that they can download with an installer.

To distribute the `.exe` file to Windows Users in a standard way, you need [INNO setup tools](https://jrsoftware.org/isdl.php). Just download the installer and feed in your files based on Wizard prompt to generate `.iss` Pascal script that can be auto-compiled into a Windows installer by Inno setup tools. You do not have to code anything, just make sure you pass my `windows.iss` to Inno setup, and it will compile the installer for you. You can pass the output to the client. Getting code signing is very troublesome, and requires payment. Not worth it for this purpose.

```bash
make windows
```

