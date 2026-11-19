# IMPORTS
import tensorflow as tf
import tf_keras as keras
import tensorflow_probability as tfp

from tf_keras.models import Sequential, Model
from tf_keras.layers import Input, Dense, BatchNormalization, Dropout
from tf_keras.optimizers import SGD, Adam
import tensorflow_probability as tfp

from ray import train

# Set seed from random number generator, for better comparisons
from numpy.random import seed
seed(123)

import matplotlib.pyplot as plt

# =======================================
# DNN related function
# =======================================
# DEEP LEARNING MODEL BUILD FUNCTION
def build_DNN(input_shape, n_hidden_layers, n_hidden_units, loss, act_fun='sigmoid', optimizer:str='sgd', learning_rate=0.01, 
            use_bn=False, use_dropout=False, use_custom_dropout=False, print_summary=False, use_variational_layer=False,
            dropout_rate=0.5, kl_weight=1e-3):
    """
    Builds a Deep Neural Network (DNN) model based on the provided parameters.
    
    Parameters:
    input_shape (tuple): Shape of the input data (excluding batch size).
    n_hidden_layers (int): Number of hidden layers in the model.
    n_hidden_units (int): Number of nodes in each hidden layer (here all hidden layers have the same shape).
    loss (keras.losses): Loss function to use in the model.
    act_fun (str, optional): Activation function to use in each layer. Default is 'sigmoid'.
    optimizer (str, optional): Optimizer to use in the model. Default is SGD.
    learning_rate (float, optional): Learning rate for the optimizer. Default is 0.01.
    use_bn (bool, optional): Whether to use Batch Normalization after each layer. Default is False.
    use_dropout (bool, optional): Whether to use Dropout after each layer. Default is False.
    use_custom_dropout (bool, optional): Whether to use a custom Dropout implementation. Default is False.
    dropout_rate (float, optional): Fraction of units dropped by the (custom) Dropout layers. Default is 0.5.
    kl_weight (float, optional): Weight of the KL term in the DenseVariational layers. Should be set to
                                 1 / (number of iterations per epoch) = batch_size / n_training_samples.
    
    Returns:
    model (Sequential): Compiled Keras Sequential model.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------      
    # Setup optimizer, depending on input parameter string
    if isinstance(optimizer, str):
        if optimizer.lower() == 'sgd':
            optimizer = SGD(learning_rate=learning_rate)
        elif optimizer.lower() == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer '{optimizer}'. Use 'sgd' or 'adam'.")
            
    # ============================================

    # Setup a sequential model
    model = Sequential()

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    # Add layers to the model, using the input parameters of the build_DNN function
    if isinstance(input_shape, int):
        input_shape = (input_shape,)

    # Add first (Input) layer, requires input shape
    model.add(Input(shape=input_shape))

    # Add remaining layers. These to not require the input shape since it will be infered during model compile
    for _ in range(n_hidden_layers):

        if use_variational_layer:
            model.add(tfp.layers.DenseVariational(units=n_hidden_units,
                                                  make_prior_fn=prior,
                                                  make_posterior_fn=posterior,
                                                  kl_weight=kl_weight,
                                                  activation=act_fun))
        else:
            model.add(Dense(units=n_hidden_units, activation=act_fun))

        # Batch normalization
        if use_bn:
            model.add(BatchNormalization())

        # Standard Dropout
        if use_dropout:
            model.add(Dropout(rate=dropout_rate))

        # Custom Dropout
        if use_custom_dropout:
            model.add(myDropout(rate=dropout_rate))

    # Add final layer
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # ============================================
    # Print model summary if requested
    if print_summary:
        model.summary() 
    
    return model

def train_DNN(config, training_config):
    '''
    Train a DNN model based on the provided configuration and data. 
    This is use in the automatic hyperparameter search and follows the format that Ray Tune expects.

    Parameters:
    config (dict): Dictionary with the configuration parameters for the model. This includes the parameters needed to build the model and can be 
                    manually set or generated by Ray Tune.
                    For convenience, the config dictionary also contains the training parameters, such as the number of epochs and batch size.
    training_config (dict): Dictionary with the training parameters, such as the number of epochs and batch size, and the data to use for training and validation (Xtrain, Ytrain, Xval, Yval).
    '''

    # A dedicated callback function is needed to allow Ray Tune to track the training process
    # This callback will be used to log the loss and accuracy of the model during training
    class TuneReporterCallback(keras.callbacks.Callback):
        """Tune Callback for Keras.
        
        The callback is invoked every epoch.
        """
        def __init__(self, logs={}):
            self.iteration = 0
            super(TuneReporterCallback, self).__init__()
    
        def on_epoch_end(self, batch, logs={}):
            self.iteration += 1
            train.report(dict(keras_info=logs, mean_accuracy=logs.get("accuracy"), mean_loss=logs.get("loss")))
    
    # --------------------------------------------  
    # === Your code here =========================
    # --------------------------------------------
    # Unpack the data tuple
    X_train, y_train, X_val, y_val = training_config["data"]

    # Training parameters
    epochs = training_config.get("epochs", config.get("epochs", 20))
    batch_size = training_config.get("batch_size", config.get("batch_size", 10000))

    # Build the model using the variables stored into the config dictionary.
    # Hint: you provide the config dictionary to the build_DNN function as a keyword argument using the ** operator.
    # Only the keys that build_DNN actually accepts are forwarded, so that epochs /
    # batch_size can safely be part of the search space as well.
    build_keys = build_DNN.__code__.co_varnames[:build_DNN.__code__.co_argcount]
    model_config = {k: v for k, v in config.items() if k in build_keys}
    model = build_DNN(**model_config)
        
    # Train the model (no need to save the history, as the callback will log the results).
    # Remember to add the TuneReporterCallback() to the list of callbacks.
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_val, y_val),
              verbose=0,
              callbacks=[TuneReporterCallback()])

    # --------------------------------------------


# CUSTOM DROPOUT IMPLEMENTATION
# Code from https://github.com/keras-team/tf-keras/issues/81
class myDropout(keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)  # Override training=True


# CUSTOM PRIOR AND POSTERIOR FUNCTIONS FOR THE VARIATIONAL LAYER
#  Code from https://keras.io/examples/keras_recipes/bayesian_neural_networks/
# The prior is defined as a normal distribution with zero mean and unit variance.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# multivariate Gaussian distribution parametrized by a learnable parameters.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


# =======================================
# PLOTTING FUNCTIONS
# =======================================

# TRAINING CURVES PLOT FUNCTION
def plot_results(history):
    """
    Plots the training and validation loss and accuracy from a Keras history object.
    Parameters:
    history (keras.callbacks.History): A History object returned by the fit method of a Keras model. 
                                       It contains the training and validation loss and accuracy for each epoch.
    Returns:
    None
    """
    
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    
    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training','Validation'])

    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training','Validation'])

    plt.show()
