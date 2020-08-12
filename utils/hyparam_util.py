import json
import tensorflow as tf
from utils.config import HYPARAM_CONFIG_FILE_PATH

def get_initializer(name):
    return name

def get_regularizer(name, l1=None, l2=None, l1l2=None):
    print(name)
    if name == 'l1':
        return tf.keras.regularizers.l1(l1)
    elif name == 'l2':
        return tf.keras.regularizers.l2(l2)
    elif(name == 'l1l2'):
        return tf.keras.regularizers.l1_l2(l1 = l1, l2 = l2)

def get_dropout_layer(rate):
    return tf.keras.layers.Dropout(rate)

def get_activation(name):
    return name

def get_optimizer(name, learning_rate):
    if name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate)
    
def load_fusion_hyparam(iteration, layer=None):
    with open(HYPARAM_CONFIG_FILE_PATH) as config_file:
        config = json.load(config_file)
        params = config[iteration]
        hyparams = {}
        if(layer == None):
            hyparams['batch_size'] = params['batch_size']
            hyparams['epochs'] = params['epochs']
            hyparams['input_width'] = params['input_width']
            hyparams['input_height'] = params['input_height']
        else:
            params = params[layer]
            # print(params)
            hyparams['kernel_initializer'] = get_initializer(**params['kernel_initializer'])
            hyparams['kernel_regularizer'] = get_regularizer(**params['kernel_regularizer'])
            hyparams['activity_regularizer'] = get_regularizer(**params['activity_regularizer'])
            hyparams['activation'] = get_activation(**params['activation'])
            hyparams['dropout_layer'] = get_dropout_layer(**params['dropout'])
            hyparams['optimizer'] = get_optimizer(**params['optimizer'])
        return hyparams