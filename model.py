import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


def build_model(Tx, LSTM_LAYER, DENSE_LAYER, RESHAPED_LAYER):
    # schema of each cell: [X_{t}, a_{t-1}, c0_{t-1}] -> RESHAPE() -> LSTM() -> DENSE()

    n_values = DENSE_LAYER.units     # input values shape
    hidden_state_dims = LSTM_LAYER.units
    
    X = Input(shape=(Tx, n_values)) 
    
    initial_hidden_state = Input(shape=(hidden_state_dims,), name='a0')
    initial_cell_state = Input(shape=(hidden_state_dims,), name='c0')
    a = initial_hidden_state
    c = initial_cell_state
    outputs = []
    
    for t in range(Tx):
        
        x = X[:, t, :]
        x = RESHAPED_LAYER(x)
        _, a, c = LSTM_LAYER(inputs=x, initial_state=[a, c])
        out = DENSE_LAYER(a)
        outputs.append(out)
        
    model = Model(inputs=[X, initial_hidden_state, initial_cell_state], outputs=outputs)
    
    return model

def music_inference_model(LSTM_LAYER, DENSE_LAYER, Ty=100):
    """
    LSTM_LAYER and DENSE_LAYER are trained, from model() 
    """
    
    n_values = DENSE_LAYER.units    # input values shape
    hidden_state_dims = LSTM_LAYER.units
    
    input_x = Input(shape=(1, n_values))
    
    initial_hidden_state = Input(shape=(hidden_state_dims,), name='a0')
    initial_cell_state = Input(shape=(hidden_state_dims,), name='c0')
    a = initial_hidden_state
    c = initial_cell_state
    x = input_x

    outputs = []
    
    for t in range(Ty):
        _, a, c = LSTM_LAYER(x, initial_state=[a, c])
        
        out = DENSE_LAYER(a)
        outputs.append(out)
 
        x = tf.math.argmax(out, -1)
        x = tf.one_hot(x, n_values)
        x = RepeatVector(1)(x)
        
    inference_model = Model(inputs=[input_x, initial_hidden_state, initial_cell_state], outputs=outputs)
    
    return inference_model

def predict_and_sample(inference_model, x_init, a_init, c_init):
    """
    Arguments:
    x_init -- np array of shape (1, 1, 90), one-hot vector initializing the values generation
    a_init -- np array of shape (1, hidden_state_dims), initializing the hidden state of the LSTM_LAYER
    c_init -- np array of shape (1, hidden_state_dims), initializing the cell state of the LSTM_LAYER
    
    Returns:
    results -- np-array of shape (Ty, 90), matrix of one-hot vectors representing the values generated
    indices -- np-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    n_values = x_init.shape[2]
    
    pred = inference_model.predict([x_init, a_init, c_init])
    indices = np.argmax(pred, axis = -1)
    results = to_categorical(indices, num_classes=n_values)
    
    return results, indices