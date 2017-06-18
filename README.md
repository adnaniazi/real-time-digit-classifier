Real-time Digit Classification using TensorFlow and PyQt
========================================================

 

This PyQt GUI uses a deep neural network classifier trained on MNIST dataset to
classify custom hand-written images in real time. The user can use mouse input
to scribble a digit and the GUI will try to classify the image as soon as the
user lifts the mouse.

 

Requirements:
-------------

1. Python 3.5.0

2. TensorFlow 1.1.0

3. Scipy 0.19.0

4. Numpy 1.12.1+mkl

5. PyQt5

6. skimage

 

How to run it?
--------------

Import all files in a project in PyCharm. Then run main.py file.

 

What is what?
-------------

1. main.py: contains all the code to to load and run the trained classifier. It
also contains code for the GUI itself.

2. model\*: These are all the files of the trained deep neural network. main.py
loads this trained classifier at startup.

3. my_gui.ui: If you want to make any changes in the GUI’s front end, this is
the file you should edit in the PyQt designer. This file is in XML format and
must be converted into Python for subsequent use in main.py file.

4. generate_py_file_ui_file.bat and my_gui.py: Clicking the bat file will
convert the my_gui.ui file into my_gui.py file. The my_gui.py file contains
Python code for GUI’s front end

5. generate_py_file_qrc_file.bat and resources.qrc file: Any icons used in the
GUI must be placed in the icons folder and these icons should then be defined in
the resources.qrc file. Then a Python file (resources_rc.py) must be generated
by running the generate_py_file_qrc_file.bat file
