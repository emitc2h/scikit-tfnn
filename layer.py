import tensorflow as tf
import numpy as np
import math


## ============================================
class Layer(object):
    """
    A class used to specify a layer in a feed-forward neural network
    """

    activations = ['relu', 'relu6', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign']

    ## --------------------------------------------
    def __init__(
        self,
        n_neurons=5,
        activation='sigmoid',
        random_seed=42
        ):
        """
        Constructor
        """

        ## layer position in the network
        self.position = None

        ## is input layer
        self.is_input  = False

        ## layer parameters from the constructor
        self.n_neurons   = n_neurons

        assert activation in self.activations, 'allowed activations are: {0}'.format(', '.join(self.activations))
        self.activation  = activation
        self.random_seed = random_seed

        ## placeholders for the tensorflow variables
        self.weights = None
        self.biases  = None
        self.output  = None


    ## --------------------------------------------
    def build(self, input_layer=None):
        """
        Builds the layer in the tensorflow session
        """

        ## Building the input layer to the network
        if input_layer is None:
            self.is_input = True
            self.output = tf.placeholder(tf.float32, shape=[None, self.n_neurons])

        ## Building any other layer
        else:
            assert not input_layer.output is None, 'Layer {0} was not built, cannot be used as input to other layer.'.format(input_layer.position)

            ## Initialize the weights
            self.weights = tf.Variable(
                tf.random_normal(
                    shape=[input_layer.n_neurons, self.n_neurons],
                    stddev=1.0/math.sqrt(self.n_neurons),
                    seed=self.random_seed
                    )
                )

            ## Initialize the biases
            self.biases = tf.Variable(
                tf.random_normal(
                    shape=[self.n_neurons],
                    stddev=1.0,
                    seed=self.random_seed
                    )
                )

            ## Define the output
            self.output = getattr(tf.nn, self.activation)(
                tf.matmul(input_layer.output, self.weights) + self.biases
                )




## ============================================
class ConvLayer(Layer):
    """
    A class to specify a convolutional layer in a feed-forward neural network
    """

    ## --------------------------------------------
    def __init__(
        self,
        img_size=(10,10),
        patch_size=(5,5),
        n_features=10,
        strides=(1,1,1,1),
        padding='SAME',
        pooling=False,
        pooling_size=(2,2),
        random_seed=42
        ):
        """
        Constructor
        """

        ## Set generic layer properties
        super(ConvLayer, self).__init__(
            n_neurons=img_size[0]*img_size[1]*n_features,
            activation='relu',
            random_seed=random_seed
            )

        ## Properties proper to a convolutional layer
        self.img_size   = img_size
        self.patch_size = patch_size
        self.n_features = n_features
        self.strides    = strides
        self.padding    = padding

        ## Properties of the pooling layer
        self.pooling      = pooling
        self.pooling_size = pooling_size

        ## placeholders for the tensorflow variables
        self.weights = None
        self.biases  = None
        self.output  = None
        self.input_image = None


    ## --------------------------------------------
    def build(self, input_layer=None):
        """
        Builds the layer in the tensorflow session
        """

        ## A few sanity checks
        assert not input_layer is None, 'You can\'t use a convolutional layer as an input layer. Use a normal layer instead and use a 1D array. It will be reshaped as an image internally.'
        assert not input_layer.output is None, 'Layer {0} was not built, cannot be used as input to other layer.'.format(input_layer.position)

        ## Reshape the image as a 2D array with placeholder dimensions
        self.input_image = tf.reshape(input_layer.output, [-1, self.img_size[0], self.img_size[1], 1])

        ## Initialize the weights
        self.weights = tf.Variable(
            tf.truncated_normal(
                shape=[self.patch_size[0], self.patch_size[1], 1, self.n_features],
                stddev=1.0/math.sqrt(self.n_neurons),
                seed=self.random_seed
                )
            )

        ## Initialize the biases
        self.biases = tf.Variable(
            tf.constant(
                0.1,
                shape=[self.n_neurons]
                )
            )

        self.conv_output = tf.nn.relu(
            tf.nn.conv2d(self.input_image, self.weights, strides=self.strides, padding=self.padding) + self.biases
            )

        if not pooling:
            self.output = self.conv_output
        else:
            self.output = getattr(tf.nn, '{0}_pool'.format(self.pooling))(self.conv_output)





