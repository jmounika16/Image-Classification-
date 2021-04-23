# Image-Classification
Our group executed an Image Classification project using Tensorflow, Python and Keras.
We are training the neural networks to recognise the image with highest accuracy possible. We are using Conventional neural network for this process.

## For our code part:
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
6. np_utils.to_categorical is used to convert an array of labeled data(from 0 to nb_classes - 1) to one-hot vector.


## Our Method:
**Data:** We imported a data set from keras by using load_data function. We normalized the data so that we don't have a very wide range. Then we had to one-hot-encode the values and for this process we used the command to_categorial() from np_utils.

**Designing our model:** For our model we used a sequential format which was previously imported from keras. Then we specified the size of the filter we need, input shape ,activation and padding. We also added the dropout value to be 0.15 initially(0.15 means it drops 15% of the existing connections). And then we added the batch normalization. It basically ensures that the network creates activations with the same distributions.This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required.

Then we added another conventional layer with same activation and padding but with increased filter size so the network can learn more complex representations.Then pooling layer with dropout and batch normalization.After adding all the required conventional layers we flattened the data. Then we created a densely connected layer by specifying the number of neurons in each layer,the numbers of neurons in succeeding layers decreases, eventually approaching the same number of neurons as there are classes in the dataset (in this case 10). The maxnorm we imported initially will be useful now to prevent overfitting.Then we added softmax activation function to select the neuron with the highest probability as its output, voting that the image belongs to that class. Then we added an optimiser and we set our no of epochs to 25.Then we compiled our model. In our next step we trained our model by passing required parameters in fit function. Then we used model.evaluate()  function and evaluate and calculate the accuracy of our model. 

**Results:** 
Initially we set the dropout to 0.15 and epochs to 25 and our model has an accuracy of 80.23%. Our goal was to increase the accuracy of this model to the maximum extent we can. So we changed the dropout and epoch values a couple of times to reach the highest accuracy possible.

**Trails**|**Dropout(%)**|**Epochs**|**Accuracy(%)**
:-----:|:-----:|:-----:|:-----:
Trail 1|0.15|25|80.23
Trail 2|0.25|30|82.1
Trail 3|0.225|25|82.17
Trail 4|0.175|25|82.42

**References:**
Explanation regarding imports: 
https://en.wikipedia.org/wiki/NumPy
https://www.tensorflow.org/
https://keras.io/

The model our group referred to:
https://stackabuse.com/image-recognition-in-python-with-tensorflow-and-keras/#:~:text=Definitions%201%20TensorFlow%2FKeras.%20Credit%3A%20commons.wikimedia.org%20TensorFlow%20is%20an,label%20for%20that%20image.%203%20Feature%20Extraction.%20

All the result values are from the jupyter notebook.
