## Image-Classification
Our group exceuted an Image Classification project using Tensorflow, Python and Keras.
We are training the neural networks to recognise the image with highest accuracy possible. We are using Conventional neural network for this process.

# For our code part:
Firstly we imported all the required libraries
#import numpy                                                                     
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
#from keras.layers.convolutional import Conv2D, MaxPooling2D
#from keras.constraints import maxnorm
#from keras.utils import np_utils

1. NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
2. A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.Sequential provides training and inference features on this model.
3. The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.
4. Keras Conv2D is a 2D Convolution Layer, this layer creates a convolution kernel that is wind with layers input which helps produce a tensor of outputs.
5. Classes from the keras.constraints module allow setting constraints (eg. non-negativity) on model parameters during training. MaxNorm weight constraint constrains the weights incident to each hidden unit to have a norm less than or equal to a desired value.
6. np_utils.to_categorical is used to convert array of labeled data(from 0 to nb_classes - 1) to one-hot vector.


# Our method
Data: we imported a data set from keras by using load_data function. We normalised the data so that we don't have a very wide range. Then we had to one-hot-encode the values and for this process we used the command to_categorial() from np_utils.

Designing our model:For our model we used sequential format which was previously imported from keras. Then we specified the size of the filter we need, input shape ,activation and padding. We also added the droupout value to be 0.2(0.2 means it drops 20% of the existing connections). And then we added the batch normalization. It basically ensures that the network creates activations with the same distributions.This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required.

Now we will be adding another conventional layer with same activation and padding but with incresed filter size so the network can learn more complex representations.
