import tensorflow as tf
# Naming convection
# A - Abhishek
HyperparamConfigA2 = {
    'iteration': 'A2',
    'layer1_kernel_intializer': 'glorot_uniform',
    'layer1_kernel_regularizer': tf.keras.regularizers.l1(0.01),
    'layer1_activity_regulazier':  tf.keras.regularizers.l2(0.01),
    'layer1_activation': 'relu',
    'layer1_dropout': tf.keras.layers.Dropout(0.2),
    'optimizer_ft': 'rmsprop',
    'optimizer_fusion': tf.keras.optimizers.RMSprop(),
    'batch_size': 8,
    'epochs': 200
}

HyperparamConfigA3 = {
    'iteration': 'A3',
    'layer1_kernel_intializer': 'glorot_uniform',
    'layer1_kernel_regularizer': None,
    'layer1_activity_regulazier':  None,
    'layer1_activation': 'relu',
    'layer1_dropout': tf.keras.layers.Dropout(0),
    'optimizer_ft': 'rmsprop',
    'optimizer_fusion': tf.keras.optimizers.RMSprop(),
    'batch_size': 8,
    'epochs': 200
}

HyperparamConfigA4 = {
    'iteration': 'A4',
    'layer1_kernel_intializer': 'glorot_uniform',
    'layer1_kernel_regularizer': None,
    'layer1_activity_regulazier':  None,
    'layer1_activation': 'relu',
    'layer1_dropout': tf.keras.layers.Dropout(0),
    'optimizer_ft': 'rmsprop',
    'optimizer_fusion': tf.keras.optimizers.RMSprop(learning_rate = 0.0001),
    'batch_size': 8,
    'epochs': 200
}

HyperparamConfigTest = {
    'iteration': 'test',
    'layer1_kernel_intializer': 'glorot_uniform',
    'layer1_kernel_regularizer': None,
    'layer1_activity_regulazier':  None,
    'layer1_activation': None,
    'layer1_dropout': tf.keras.layers.Dropout(0),
    'optimizer_ft': 'rmsprop',
    'optimizer_fusion': tf.keras.optimizers.RMSprop(learning_rate = 0.0001),
    'batch_size': 8,
    'epochs': 1
}

HyperparamConfigD1 = {
    'iteration': 'D1',
    'layer1_kernel_intializer': 'glorot_uniform',
    'layer1_kernel_regularizer': None,
    'layer1_activity_regulazier':  None,
    'layer1_activation': 'relu',
    'layer1_dropout': tf.keras.layers.Dropout(0.2),
    'optimizer_ft': 'rmsprop',
    'optimizer_fusion': tf.keras.optimizers.RMSprop(learning_rate = 0.0001),
    'batch_size': 8,
    'epochs': 200
}
