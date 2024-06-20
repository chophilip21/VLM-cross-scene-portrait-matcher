# Comparison of framework

The goal of this repository is to create a standalone ML application that can be packaged and delivered to clients using Mac or Windows. I have spent time tying to make things work with Toga, but despite the ease of building UI with the framework, it had major bug in terms of interaction with multiprocessing modules when packaged for Windows:


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


# PyQT vs Pyside

1. Licensing Differences
PyQt6: Developed by Riverbank Computing, PyQt is available under the GPL (General Public License) and a commercial license. If you want to use PyQt in a proprietary application without releasing your source code, you need to purchase a commercial license.
PySide6: Also known as Qt for Python, PySide is developed by the Qt Company and is available under the LGPL (Lesser General Public License) as well as a commercial license. The LGPL allows for use in proprietary applications under certain conditions without the need for a commercial license.
2. Maintenance and Support
PyQt6: Riverbank Computing maintains PyQt. They offer commercial support for businesses that need it.
PySide6: Maintained by the Qt Company, PySide is an official part of the Qt ecosystem. This means it gets direct updates and support from the same organization that develops the Qt framework itself.
3. API and Compatibility
While both bindings aim to provide similar functionality, there may be slight differences in their APIs or in how they implement certain features. Developers might prefer one over the other based on specific needs or existing infrastructure.
4. Historical Context
PyQt has been around longer and was the primary way to use Qt with Python for many years.
PySide was created later by the Qt Company to offer an alternative with different licensing terms, making it more accessible for certain types of projects.
5. Ecosystem and Community Preferences
Some developers and companies have historical preferences or dependencies on one binding over the other based on their project history, community support, or specific technical advantages they might find in one.
In summary, the existence of both PyQt and PySide provides developers with options, especially regarding licensing and support. This flexibility allows developers to choose the tool that best fits their project requirements and legal constraints.

# Why would anyone use PyQT then?

1. Maturity and Stability
Established History: PyQt has been around longer than PySide and has a long track record of stability and reliability.
Community Support: A more established community can mean more available resources, such as tutorials, examples, and forums for troubleshooting.
2. Feature Completeness
API Coverage: Some developers feel that PyQt6 provides more complete and robust API coverage for certain Qt features compared to PySide6.
Updates: There are instances where PyQt might get certain features or updates faster than PySide due to differences in development cycles.
3. Performance and Compatibility
Performance: In some cases, developers have noted differences in performance or compatibility with certain systems or Qt versions, preferring PyQt for their specific use cases.
Compatibility: Some projects started with PyQt and switching to PySide might introduce compatibility issues that require significant refactoring.
4. Licensing Considerations
Commercial License: For some organizations, having a commercial license with dedicated support and guaranteed bug fixes can be crucial. Riverbank Computing offers commercial support for PyQt, which can be a deciding factor for enterprise projects.
Legal Simplicity: While LGPL is permissive, some companies prefer the clarity and simplicity of the commercial license provided by PyQt to avoid potential legal pitfalls associated with open-source licenses.
5. Existing Codebase
Legacy Projects: Projects that started with PyQt might find it easier to continue with it rather than porting the codebase to PySide, which could introduce new bugs or require significant changes.
6. Support and Documentation
Commercial Support: Riverbank Computing offers commercial support for PyQt, which can be invaluable for enterprise projects needing guaranteed assistance and timely updates.
Documentation: Some developers might prefer the style or structure of PyQt’s documentation and find it easier to navigate.
7. Personal or Organizational Preference
Familiarity: Developers or teams familiar with PyQt might prefer to stick with it due to personal comfort and familiarity.
Project Requirements: Specific project requirements or constraints might make PyQt a more suitable choice despite the cost.
In conclusion, while PySide6 being free under LGPL is a significant advantage, there are multiple factors such as maturity, stability, performance, support, and licensing preferences that might lead developers or organizations to choose PyQt6 instead.