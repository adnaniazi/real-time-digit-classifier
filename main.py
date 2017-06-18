# import all QT GUI functionality
from PyQt5.QtGui import *

# import functionality for QT GUI widgets
# QObject is a class of QtCore
from PyQt5.QtCore import *

# import the main class of our UI designed in QT designer
from my_gui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QApplication

#scribble imports
from PyQt5.QtCore import QDir, QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QImageWriter, QPainter, QPen, qRgb
from PyQt5.QtWidgets import (QAction, QColorDialog, QFileDialog,
        QInputDialog, QMenu, QMessageBox, QWidget)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

# DL imports
from skimage import util
import os
import numpy as np
from scipy.misc import imread
import tensorflow as tf
import scipy
from scipy.misc import imsave

class MyMainGui(QMainWindow, Ui_MainWindow):

    # extend constructors of QMainWindow and Ui_mainWindow
    def __init__(self, parent=None):
        super(MyMainGui, self).__init__()
        # call setupUi function in Ui_MainWindow class
        self.setupUi(self)
        # start writing your custom code here
        self.scribbleArea = ScribbleArea()
        self.consoleText = 'Application started'
        self.textEdit_console.setText(self.consoleText)

        # DL stuff
        # To stop potential randomness
        seed = 128
        rng = np.random.RandomState(seed)

        # The first step is to set directory paths, for safekeeping!
        root_dir = os.path.abspath('./')
        data_dir = os.path.join(root_dir, 'data')
        sub_dir = os.path.join(root_dir, 'sub')

        # number of neurons in each layer
        input_num_units = 28*28
        hidden_num_units = 500
        output_num_units = 10

        # define placeholders
        x = tf.placeholder(tf.float32, [None, input_num_units])
        y = tf.placeholder(tf.float32, [None, output_num_units])

        # set remaining variables
        epochs = 5
        batch_size = 128
        learning_rate = 0.01

        weights = {
            'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
            'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
        }

        biases = {
            'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
            'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
        }

        # Now create our neural networks computational graph
        hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
        hidden_layer = tf.nn.relu(hidden_layer)

        output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

        # Also, we need to define cost of our neural network
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

        #And set the optimizer, i.e. our backpropogation algorithm.
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # initialize all the variables
        init = tf.global_variables_initializer()

        sess = tf.Session()
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # Restore variables from disk.
        saver.restore(sess, "model")
        print("Model restored.")
        self.sendToConsole('Trained model restored.')

        predict = tf.argmax(output_layer, 1)

        # make the scribble_widget area parent of the scribbleArea
        # so that the scribleArea fits inside the area we set in designer
        self.scribbleArea.setParent(self.widget_scribble)
        self.scribbleArea.resize(400, 400)

        # clear the scribble when clear button is pressed
        self.pushButton_clear.clicked.connect(self.scribbleArea.clearImage)
        # Display image and classify it
        self.root_dir = root_dir
        self.predict = predict
        self.x = x
        self.sess = sess
        self.scribbleArea.trigger.connect(self.imageReady)

        self.historyText = ''
        self.textEdit_classificationHistory.setFontPointSize(20)

    def imageReady(self):
        self.sendToConsole('Mouse released')
        predict = self.predict
        x = self.x
        sess = self.sess
        root_dir = self.root_dir

        self.sendToConsole('Image ready')
        image = QPixmap('img.png')
        h = self.label_classifierInputImage.height()
        w = self.label_classifierInputImage.width()
        self.label_classifierInputImage.setPixmap(image.scaled(w,h))

        with sess.as_default():
            #load picture to be classified
            temp=[]
            image_path = os.path.join(root_dir, 'img.png')
            img = imread(image_path, flatten=True)
            img = img.astype('float32')
            img_28 = scipy.misc.imresize(img, (28, 28), interp='bilinear', mode=None)
            img_inv = util.invert(img_28) #invert_color
            temp.append(img_inv)
            test_img = np.stack(temp)
            self.sendToConsole('Classifying image...')
            pred = predict.eval({x: test_img.reshape(-1, 784)})
            self.sendToConsole('Classifier output: ' + str(pred) +'\n')

            # save inverse 28x28image
            image_inv28_save_path = os.path.join(root_dir, 'img28_inv.png')
            imsave(image_inv28_save_path, img_inv)

            # Load inverse 28x28image into the GUI
            image = QPixmap('img28_inv.png')
            h = self.label_classifierInputImage.height()
            w = self.label_classifierInputImage.width()
            self.label_classifierInputImage_downSampled.setPixmap(image.scaled(w,h))

            # Update the predict in the GUI
            prediction = np.asscalar(pred)
            self.label_classifiedDigit.setText(str(prediction))

            # Update the classifier history stream
            self.historyText = self.historyText + str(prediction)
            self.textEdit_classificationHistory.setText(self.historyText)
            self.textEdit_classificationHistory.setFocus()
            self.textEdit_classificationHistory.moveCursor(QTextCursor.End)

    def sendToConsole(self, string):
        self.consoleText = self.consoleText + '\n' + string
        self.textEdit_console.setText(self.consoleText)
        self.textEdit_console.moveCursor(QTextCursor.End)



class ScribbleArea(QWidget, QObject):
    trigger = pyqtSignal()

    def __init__(self, parent=None):
        super(ScribbleArea, self).__init__(parent)
        self.setAttribute(Qt.WA_StaticContents)
        self.modified = False
        self.scribbling = False
        self.myPenWidth = 40
        self.myPenColor = Qt.black
        self.image = QImage()
        self.lastPoint = QPoint()


    def openImage(self, fileName):
        loadedImage = QImage()
        if not loadedImage.load(fileName):
            return False

        newSize = loadedImage.size().expandedTo(self.size())
        self.resizeImage(loadedImage, newSize)
        self.image = loadedImage
        self.modified = False
        self.update()
        return True

    def saveImage(self, fileName, fileFormat):
        visibleImage = self.image
        self.resizeImage(visibleImage, self.size())

        if visibleImage.save(fileName, fileFormat):
            self.modified = False
            return True
        else:
            return False

    def setPenColor(self, newColor):
        self.myPenColor = newColor

    def setPenWidth(self, newWidth):
        self.myPenWidth = newWidth

    def clearImage(self):
        self.image.fill(qRgb(255, 255, 255))
        self.modified = True
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.scribbling = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.scribbling:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.scribbling:
            self.drawLineTo(event.pos())
            self.scribbling = False
            print("mouse released")

            visibleImage = self.image
            visibleImage.save('img.png', 'PNG')
            self.clearImage()
            self.trigger.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        dirtyRect = event.rect()
        painter.drawImage(dirtyRect, self.image, dirtyRect)
        print("paint event")

    def resizeEvent(self, event):
        if self.width() > self.image.width() or self.height() > self.image.height():
            newWidth = max(self.width() + 0, self.image.width())
            newHeight = max(self.height() + 0, self.image.height())
            self.resizeImage(self.image, QSize(newWidth, newHeight))
            self.update()

        super(ScribbleArea, self).resizeEvent(event)
        print("resize event")

    def drawLineTo(self, endPoint):
        painter = QPainter(self.image)
        painter.setPen(QPen(self.myPenColor, self.myPenWidth, Qt.SolidLine,
                Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(self.lastPoint, endPoint)
        self.modified = True

        rad = self.myPenWidth / 2 + 2
        self.update(QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))
        self.lastPoint = QPoint(endPoint)

    def resizeImage(self, image, newSize):
        if image.size() == newSize:
            return

        newImage = QImage(newSize, QImage.Format_RGB32)
        newImage.fill(qRgb(255, 255, 255))
        painter = QPainter(newImage)
        painter.drawImage(QPoint(0, 0), image)
        self.image = newImage

    def print_(self):
        printer = QPrinter(QPrinter.HighResolution)

        printDialog = QPrintDialog(printer, self)
        if printDialog.exec_() == QPrintDialog.Accepted:
            painter = QPainter(printer)
            rect = painter.viewport()
            size = self.image.size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.image.rect())
            painter.drawImage(0, 0, self.image)
            painter.end()

    def isModified(self):
        return self.modified

    def penColor(self):
        return self.myPenColor

    def penWidth(self):
        return self.myPenWidth



if __name__ == "__main__":
    app = QApplication([])
    my_gui = MyMainGui()
    my_gui.show()
    app.exit(app.exec_())