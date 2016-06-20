import tensorflow as tf
import numpy as np

from layer import Layer

## --------------------------------------------
def one_hot_vector(y):
    """
    Takes a array containing a categorical
    variable and return an array of one-hot vector
    Assumes there are n categorical values
    ordered like 0, 1, ..., n-1
    """

    n_categories = np.unique(y).shape[0]
    y_one_hot    = np.zeros((y.shape[0], n_categories))

    for i in range(y.shape[0]):
        try:
            y_one_hot[i, y[i]] = 1
        except IndexError:
            pass

    return y_one_hot.astype(int)




## ============================================
class NeuralNetwork(object):
    """
    A Neural network class with a scikit-compliant interface
    """

    cost_functions      = ['log-likelihood', 'cross-entropy']
    regularizations     = ['l1', 'l2', 'none']
    learning_algorithms = ['Adam', 'GradientDescent', 'Adagrad', 'Ftrl', 'RMSProp']


    ## --------------------------------------------
    def __init__(
        self,
        hidden_layers = [],
        learning_algorithm='Adam',
        output_activation='softmax',
        cost_function='cross-entropy',
        regularization='none',
        reg_lambda=1.0,
        learning_rate=0.1,
        early_stopping=False,
        stagnation=10,
        target_accuracy=None,
        n_epochs=10,
        mini_batch_size=10,
        ):
        """
        Constructor
        """

        ## List of Layer objects, the input and output layers are automatically specified
        ## using the input data and the output_activation parameter
        self.hidden_layers = hidden_layers

        ## Parameters from the constructor
        assert learning_algorithm in self.learning_algorithms, 'Available learning algorithms are {0}'.format(', '.join(self.learning_algorithms))
        self.learning_algorithm = learning_algorithm

        assert cost_function in self.cost_functions, 'Available cost functions are {0}'.format(', '.join(self.cost_functions))
        self.cost_function      = cost_function
        self.output_activation  = output_activation

        assert regularization in self.regularizations, 'Available regularizations are {0}'.format(', '.join(self.regularizations))
        self.regularization     = regularization
        self.reg_lambda         = reg_lambda

        self.learning_rate      = learning_rate

        self.early_stopping     = early_stopping
        self.stagnation         = stagnation
        self.target_accuracy    = target_accuracy

        self.n_epochs           = n_epochs
        self.mini_batch_size    = mini_batch_size

        ## functions to keep around
        self.correct_prediction = None
        self.accuracy           = None
        self.accuracy_buffer    = []
        self.current_epoch      = 0 




    ## --------------------------------------------
    def build(self, X, y):
        """
        Builds the neural network in tensorflow
        """

        ## Start a tensorflow interactive session
        self.session = tf.InteractiveSession()

        ## First, create a placeholder for the targets
        self.targets = tf.placeholder(tf.float32, shape=[None, self.n_categories])

        ## First, create the input layer
        self.input_layer = Layer(n_neurons=self.n_features)
        self.input_layer.build()

        ## Then create all the hidden layers
        current_input_layer = self.input_layer

        for layer in self.hidden_layers:
            layer.build(current_input_layer)
            current_input_layer = layer

        ## Create the output layer
        self.output_layer = Layer(n_neurons=self.n_categories, activation=self.output_activation)
        self.output_layer.build(current_input_layer)

        ## Define the cost function
        self.cost = None
        if self.cost_function == 'log-likelihood':

            self.cost = tf.reduce_mean(-tf.log(1e-37 + tf.reduce_sum(self.targets * self.output_layer.output, reduction_indices=[1])))
        else: ## cross-entropy
            self.cost = tf.reduce_mean(-tf.reduce_sum(self.targets * tf.log(1e-37 + self.output_layer.output), reduction_indices=[1]))

        ## Define the regularization parameters and function
        self.reg_lambda_param = tf.placeholder(tf.float32)
        self.batch_size       = tf.placeholder(tf.float32)

        if self.regularization == 'l1':
            self.reg_term = tf.reduce_sum(tf.abs(self.output_layer.weights))
            for layer in self.hidden_layers:
                self.reg_term += tf.reduce_sum(tf.abs(layer.weights))

        elif self.regularization == 'l2':
            self.reg_term = tf.reduce_sum(self.output_layer.weights * self.output_layer.weights)
            for layer in self.hidden_layers:
                self.reg_term += tf.reduce_sum(layer.weights * layer.weights)

        else:
            self.reg_term = None

        ## Add the regularization term to the cost function
        if self.reg_term is None:
            self.reg_cost = self.cost
        else:
            self.reg_cost = self.cost + (self.reg_lambda_param/(2*self.batch_size))*self.reg_term

        ## Define the train step
        self.train_step = getattr(tf.train, '{0}Optimizer'.format(self.learning_algorithm))(self.learning_rate).minimize(self.reg_cost)

        ## Initialize everything
        self.session.run(tf.initialize_all_variables())




    ## -----------------------------------------
    def create_mini_batches(self, X, y):
        """
        Creates a list of mini-batches for stochastic
        gradient descent
        """
    
        data = np.hstack([X, y])
    
        np.random.shuffle(data)
        shuffled_X = data[:,:self.n_features]
        shuffled_y = data[:,self.n_features:]
    
        n_batches = shuffled_y.shape[0]/self.mini_batch_size
    
        return [(shuffled_X[i*self.mini_batch_size : (i+1)*self.mini_batch_size], shuffled_y[i*self.mini_batch_size : (i+1)*self.mini_batch_size]) for i in xrange(n_batches)]



    ## --------------------------------------------
    def prepare_fit(self, X, y, verbose=False, val_X=None, val_y=None):
        """
        A few steps to prepare for the fit
        """

        ## Reset number of epochs
        self.current_epoch = 0

        ## Make sure input variables and target are compatible

        ## Do some preprocessing on the input data
        self.n_features = X.shape[1]

        ## Turn the output into a one-hot vector
        if len(y.shape) < 2:
            y_one_hot = one_hot_vector(y)
        else:
            y_one_hot = y

        self.n_categories = y_one_hot.shape[1]


        ## Build the tensorflow variables and functions
        self.build(X, y_one_hot)

        ## Build an accuracy function given that a validation set is provided
        val_provided = False
        if (not val_X is None) and (not val_y is None):
            val_provided = True

        if val_provided:

            if len(val_y.shape) < 2:
                val_y_one_hot = one_hot_vector(val_y)
            else:
                val_y_one_hot = val_y

            self.correct_prediction = tf.equal(tf.argmax(self.output_layer.output, 1), tf.argmax(self.targets, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.accuracy_buffer = []

        return y_one_hot, val_y_one_hot



    ## --------------------------------------------
    def epoch(self, X, y_one_hot, verbose=False, val_X=None, val_y_one_hot=None):
        """
        train for a single epoch
        """

        current_accuracy = 0.0

        if self.current_epoch > self.n_epochs:
            return current_accuracy, False

        val_provided = False
        if (not val_X is None) and (not val_y_one_hot is None):
            val_provided = True

        ## Calculate accuracy on validation sample
        if val_provided:

            ## Construct the feed_dict
            feed_dict={self.input_layer.output : val_X, self.targets : val_y_one_hot}
            for layer in self.hidden_layers:
                if hasattr(layer, 'dropout_rate'):
                    feed_dict[layer.keep_prob] = 1.0

            current_accuracy = self.session.run(self.accuracy, feed_dict=feed_dict)


            ## Print out epoch, accuracy
            if verbose:
                print 'Epoch {0}, validation sample accuracy: {1}'.format(self.current_epoch, current_accuracy)
            else:
                print 'Epoch {0} ...'.format(self.current_epoch)


            if not self.target_accuracy is None:
                if current_accuracy >= self.target_accuracy:
                    return current_accuracy, False

            ## Fill in the accuracy buffer
            self.accuracy_buffer.append(current_accuracy)
            if len(self.accuracy_buffer) > self.stagnation:
                self.accuracy_buffer.pop(0)

            ## Estimate accuracy change on the validation sample
            if self.current_epoch > 0:
                lin_reg_params = np.polyfit(range(len(self.accuracy_buffer)), self.accuracy_buffer, 1)
                rel_accuracy_change = lin_reg_params[0]/(1.0 - self.accuracy_buffer[0])
                if self.early_stopping:
                    if rel_accuracy_change < 0.00001:
                        return current_accuracy, False


        batches = self.create_mini_batches(X, y_one_hot)

        for batch in batches:

            ## Construct the feed_dict
            feed_dict={
                self.input_layer.output : batch[0],
                self.targets            : batch[1],
                self.reg_lambda_param   : self.reg_lambda,
                self.batch_size         : self.mini_batch_size
            }
            for layer in self.hidden_layers:
                if hasattr(layer, 'dropout_rate'):
                    feed_dict[layer.keep_prob] = layer.dropout_rate

            self.train_step.run(
                feed_dict=feed_dict
            )

        self.current_epoch += 1

        return current_accuracy, True



    ## --------------------------------------------
    def fit(self, X, y, verbose=False, val_X=None, val_y=None):
        """
        fits the model to the training data
        """

        y_one_hot, val_y_one_hot = self.prepare_fit(X, y, verbose, val_X, val_y)

        status = True

        while(status):
            _, status = self.epoch(X, y_one_hot, verbose, val_X, val_y_one_hot)




    ## --------------------------------------------
    def predict_proba(self, X):
        """
        returns probabilities for each category, for each sample provided
        """

        feed_dict={self.input_layer.output: X}
        for layer in self.hidden_layers:
                if hasattr(layer, 'dropout_rate'):
                    feed_dict[layer.keep_prob] = 1.0

        return self.output_layer.output.eval(feed_dict=feed_dict)


