# What problem is it trying to solve?

This repository presents a framework for robust cross-scene subject correspondence by leveraging a pretrained vision-language model to propose candidate individuals in crowded images, extracting face embeddings for both scenes, and iteratively matching and filtering these embeddings to align subjects from a noisy, unordered set of group photos to a reference set of single‐subject portraits

To give you more detail, 

- Assume there are two groups of portrait photos taken at two different scenes---A) Crowded graduation stage, where there are many subjects B) Behind the stage, where there is only one subject.
- Scene A) and B) have n and m number of photos. The problem that it solves is trying to find matching subjects in two scenes for every single photo, using B) as the reference.
- Both A and B are in complete random order. Scene A has a lot of noise. It also contains random images that aren't matching with any subject and scene B.
- Using pretrained Kosmos-2 VLM, model, the most important subject (i.e the graduating student) is identified from scene A, and two other possible candidates.
- Face detection and recognition model runs on the three candidates captured in a single photo of scene A. All subjects on Scene B are converted into face vectors.
- Iterate over all possible combination and group face vectors in both scenes. Remove face vectors that appear a lot, because these are most likely photos of teachers.      

| ![Alt 1](https://tamupvfa.b-cdn.net/app/uploads/2023/12/web20231214_PVFA_Commencement_AS_0078-1024x683.jpg) | ![Alt 2](https://gsrstudio.ca/wp-content/uploads/2023/06/grads-photos-Toronto-2023.jpg) |
|:----------------------------:|:----------------------------:|
| *Example of image from group A*                  | *Example of image from group B                  |





# Packaging to executables

The front-end of the application is designed with `Pyside6`. It can be deployed into desktop application using:

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

