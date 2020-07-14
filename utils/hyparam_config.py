import tensorflow as tf
HyperparamConfig2 = {
    'iteration': 2,
    'layer1_kernel_intializer': 'glorot_uniform',
    'layer1_kernel_regularizer': tf.keras.regularizers.l1(0.01),
    'layer1_activity_regulazier':  tf.keras.regularizers.l2(0.01),
    'layer1_activation': 'relu',
    'layer1_dropout': tf.keras.layers.Dropout(0.2),
    'optimizer': 'rmsprop'
}

HyperparamConfig3 = {
    'iteration': 3,
    'layer1_kernel_intializer': 'glorot_uniform',
    'layer1_kernel_regularizer': None,
    'layer1_activity_regulazier':  None,
    'layer1_activation': 'relu',
    'layer1_dropout': tf.keras.layers.Dropout(0),
    'optimizer': 'rmsprop'
}