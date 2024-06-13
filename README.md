# Comparison of framework

The goal of this repository is to create a standalone ML application that can be packaged and delivered to clients using Mac or Windows. 

I have spent time tying to make things work with Toga, but despite the ease of building UI with the framework, it had major bug in terms of interaction with multiprocessing modules when packaged for Windows:


| Feature                     | PyQt                               | PySide2                            | PySide6                            | Kivy                                | Tkinter                            | WxPython                          | Toga                               |
|-----------------------------|------------------------------------|------------------------------------|------------------------------------|-------------------------------------|------------------------------------|-----------------------------------|------------------------------------|
| **Ease of Use**             | Moderate                           | Moderate                           | Moderate                           | Moderate                            | Easy                               | Moderate                          | Easy                               |
| **Documentation**           | Excellent                          | Good                               | Good                               | Good                                | Good                               | Good                              | Fair                               |
| **Community Support**       | Large                              | Growing                            | Small but growing                  | Growing                             | Large                              | Moderate                          | Small                              |
| **Cross-Platform Support**  | Yes                                | Yes                                | Yes                                | Yes                                 | Yes                                | Yes                               | Yes                                |
| **Maturity**                | High                               | Moderate                           | Newer but stable                   | Moderate                            | High                               | High                              | Low                                |
| **Packaging**               | Good (with PyInstaller or PyQt's tools) | Good (with PyInstaller or fbs)     | Good (with PyInstaller or fbs)     | Moderate (with Buildozer for mobile) | Good (with PyInstaller or cx_Freeze) | Good (with PyInstaller)          | Good (with Briefcase)              |
| **Performance**             | High                               | High                               | Improved                           | High                                | Moderate                           | High                              | Moderate                           |
| **Multiprocessing/Threading**| Good                               | Good                               | Good                               | Good                                | Limited                            | Good                              | Limited                            |
| **Look and Feel**           | Native                             | Native                             | Native                             | Customizable, but non-native        | Basic, native                      | Native                            | Customizable, but non-native       |
| **Learning Curve**          | Steep                              | Steep                              | Steep                              | Moderate                            | Gentle                             | Moderate                          | Gentle                             |
| **License**                 | GPL/commercial                     | LGPL                               | LGPL                               | MIT                                 | BSD                                | LGPL                              | BSD                                |

Pyside2 (LGPL license) based on QT5 was recommended, but now that Pyside6 based on QT6 is available, I am going to try that out first.

| Feature                       | PySide2                              | PySide6                              |
|-------------------------------|--------------------------------------|--------------------------------------|
| **Qt Version**                | Qt 5                                 | Qt 6                                 |
| **Stability**                 | Stable, well-tested                  | Newer, evolving but stable           |
| **Features**                  | Comprehensive, established           | New features from Qt 6               |
| **Performance**               | Optimized for Qt 5                   | Improved performance with Qt 6       |
| **Support for New Technologies** | Limited to Qt 5 capabilities        | Access to latest Qt technologies     |
| **Documentation**             | Good, extensive                      | Good, growing                        |
| **Community Support**         | Growing                              | Small but growing                    |
| **Backward Compatibility**    | Compatible with Qt 5                 | Not backward compatible with Qt 5    |
| **License**                   | LGPL                                 | LGPL                                 |


## Getting started with Pyinside6

```bash
python -m venv env
#linux
source env/bin/activate
#windows
env\Scripts\activate.bat
pip install pyside6
```