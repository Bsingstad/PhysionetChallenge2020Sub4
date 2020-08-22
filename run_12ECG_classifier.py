#!/usr/bin/env python
import numpy as np, os, sys, joblib
import joblib
import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat



def res_block(X):
  X_shortcut = X
  X_shortcut = keras.layers.MaxPool1D(pool_size=1)(X_shortcut)

  X = keras.layers.BatchNormalization()(X)
  X = keras.layers.Activation("relu")(X)
  X = keras.layers.Dropout(0.2)(X)
  X = keras.layers.Conv1D(filters=12, kernel_size=5, activation="relu", padding="same")(X)
  X = keras.layers.BatchNormalization()(X)
  X = keras.layers.Activation("relu")(X)
  X = keras.layers.Dropout(0.2)(X)
  X = keras.layers.Conv1D(filters=12, kernel_size=5, activation="relu", padding="same")(X)
  X = keras.layers.add([X,X_shortcut])
  return X


def create_model():

    # define two sets of inputs
    inputA = keras.layers.Input(shape=(5000,12)) 
    inputB = keras.layers.Input(shape=(2,))
    # the first branch operates on the first input
    mod1 = keras.layers.Conv1D(filters=12, kernel_size=5, activation="relu",input_shape=(10000,12),padding="same")(inputA)
    mod1 = keras.layers.BatchNormalization()(mod1)
    mod1 = keras.layers.Activation("relu")(mod1)

    mod1_shortcut1 = mod1
    mod1_shortcut1 = keras.layers.MaxPool1D(pool_size=1)(mod1_shortcut1)

    mod1 = keras.layers.Conv1D(filters=12, kernel_size=5, activation="relu", padding="same")(mod1)
    mod1 = keras.layers.BatchNormalization()(mod1)
    mod1 = keras.layers.Activation("relu")(mod1)
    mod1 = keras.layers.Dropout(0.2)(mod1)
    mod1 = keras.layers.Conv1D(filters=12, kernel_size=5, activation="relu", padding="same")(mod1)
    mod1 = keras.layers.add([mod1,mod1_shortcut1])
    for i in range(2):
        mod1 = res_block(mod1)


    mod1 = keras.layers.BatchNormalization()(mod1)
    mod1 = keras.layers.Activation("relu")(mod1)
    #mod1 = keras.layers.add([mod1,mod1_shortcut])
    mod1 = keras.layers.LSTM(100)(mod1)
    #mod1 = keras.layers.Flatten()(mod1)
    mod1 = keras.layers.Dense(100, activation='relu')(mod1) #sigmoid -> relu test
    mod1 = keras.Model(inputs=inputA, outputs=mod1)
    #mod1 = keras.layers.add([mod1,mod1_shortcut])
    # the second branch opreates on the second input
    mod2 = keras.layers.Dense(2, activation="relu")(inputB) # icrease 2 ->10
    mod2 = keras.Model(inputs=inputB, outputs=mod2)
    # combine the output of the two branches
    combined = keras.layers.concatenate([mod1.output, mod2.output])
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    
    z = keras.layers.Dense(27, activation="sigmoid")(combined)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = keras.Model(inputs=[mod1.input, mod2.input], outputs=z)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.BinaryAccuracy(
        name='accuracy', dtype=None, threshold=0.5),tf.keras.metrics.Recall(name='Recall'),tf.keras.metrics.Precision(name='Precision'), 
                        tf.keras.metrics.AUC(
            num_thresholds=200,
            curve="ROC",
            summation_method="interpolation",
            name="AUC",
            dtype=None,
            thresholds=None,
            multi_label=True,
            label_weights=None,
        )])

    #@title Plot model for better visualization
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model



def run_12ECG_classifier(data,header_data,loaded_model):
    


    threshold = np.array([0.12820681, 0.06499375, 0.13454682, 0.16845625, 0.1470617 ,
    0.2161416 , 0.16106858, 0.1051053 , 0.16673433, 0.21358207,
    0.17808011, 0.05360209, 0.0879685 , 0.06232401, 0.11914249,
    0.00379602, 0.15083605, 0.20306677, 0.15644205, 0.13406455,
    0.17194449, 0.11921279, 0.21419376, 0.16725275, 0.17113625,
    0.08283495, 0.09289312])


    # Use your classifier here to obtain a label and score for each class.
    model = loaded_model
    padded_signal = keras.preprocessing.sequence.pad_sequences(data, maxlen=5000, truncating='post',padding="post")
    reshaped_signal = padded_signal.reshape(1,5000,12)

    gender = header_data[14][6:-1]
    age=header_data[13][6:-1]
    if gender == "Male":
        gender = 0
    elif gender == "male":
        gender = 0
    elif gender =="M":
        gender = 0
    elif gender == "Female":
        gender = 1
    elif gender == "female":
        gender = 1
    elif gender == "F":
        gender = 1
    elif gender =="NaN":
        gender = 2

    # Age processing - replace with nicer code later
    if age == "NaN":
        age = -1
    else:
        age = int(age)

    demo_data = np.asarray([age,gender])
    reshaped_demo_data = demo_data.reshape(1,2)

    combined_data = [reshaped_signal,reshaped_demo_data]


    score  = model.predict(combined_data)[0]
    
    binary_prediction = score > threshold
    binary_prediction = binary_prediction * 1
    classes = ['10370003','111975006','164889003','164890007','164909002','164917005','164934002','164947007','17338001',
 '251146004','270492004','284470004','39732003','426177001','426627000','426783006','427084000','427172004','427393009','445118002','47665007','59118001',
 '59931005','63593006','698252002','713426002','713427006']

    return binary_prediction, score, classes

def load_12ECG_model(model_input):
    model = create_model()
    f_out='model.h5'
    filename = os.path.join(model_input,f_out)
    model.load_weights(filename)

    return model
