#install scikit-image using pip whl
from skimage import util
import os
import numpy as np
from scipy.misc import imread
import tensorflow as tf
import scipy

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)
print(rng)
developing = False


# The first step is to set directory paths, for safekeeping!
root_dir = os.path.abspath('./')
data_dir = os.path.join(root_dir, 'data')
sub_dir = os.path.join(root_dir, 'sub')

# Now comes the main part! Let us define our neural network architecture.
# We define a neural network with 3 layers;  input, hidden and output.
# The number of neurons in input and output are fixed, as the input is
# our 28 x 28 image and the output is a 10 x 1 vector representing the class.
# We take 500 neurons in the hidden layer. This number can vary according
# to your need. We also assign values to remaining variables.


### set all variables

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

### define weights and biases of the neural network
# refer https://www.analyticsvidhya.com/blog/2016/03/introduction-deep-learning-fundamentals-neural-networks/
# article if you don't understand the terminologies)

weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

#Now create our neural networks computational graph
hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

# Also, we need to define cost of our neural network
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

#And set the optimizer, i.e. our backpropogation algorithm.
# Here we use Adam, which is an efficient variant of
# Gradient Descent algorithm. There are a number of other optimizers
# available in tensorflow
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#After defining our neural network architecture, let's
# initialize all the variables
init = tf.global_variables_initializer()

# Now let us create a session, and run our neural network
# in the session. We also validate our models accuracy on
# validation set that we created
# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    # Restore variables from disk.
    saver.restore(sess, "model")
    print("Model restored.")


    # find predictions on val set
    #pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    #accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    #print("Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)}))

    predict = tf.argmax(output_layer, 1)

    #load picture to be classified
    temp=[]
    image_path = os.path.join(root_dir, 'img.png')
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    img_28 = scipy.misc.imresize(img, (28, 28), interp='bilinear', mode=None)
    img_inv = util.invert(img_28) #invert_colors
    #temp.append(img)
    #test_img = np.stack(temp)

    # pylab.imshow(img_inv, cmap='gray')
    # pylab.axis('off')
    # pylab.show(block=False)

    temp.append(img_inv)
    test_img = np.stack(temp)

    print(test_img)
    pred = predict.eval({x: test_img.reshape(-1, 784)})
    print('pred', pred)

#
# # To test our model with our own eyes, let's visualize its predictions
# img_name = rng.choice(test.filename)
# filepath = os.path.join(data_dir, 'Train', 'Images', 'test', img_name)
#
# img = imread(filepath, flatten=True)
#
# test_index = int(img_name.split('.')[0]) - 49000
#
# print("Prediction is: ", pred[test_index])
#
# pylab.imshow(img, cmap='gray')
# pylab.axis('off')
# pylab.show()