#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np, os, sys, joblib
from scipy.io import loadmat





def train_12ECG_classifier(input_directory, output_directory):
    # Load data.
    print('Loading data...')

    gender=[]
    age=[]
    labels=[]
    ecg_filenames=[]
    for ecgfilename in sorted(os.listdir(input_directory)):
        if ecgfilename.endswith(".mat"):
            data, header_data = load_challenge_data(input_directory+"/"+ecgfilename)
            labels.append(header_data[15][5:-1])
            ecg_filenames.append(input_directory + "/" + ecgfilename)
            gender.append(header_data[14][6:-1])
            age.append(header_data[13][6:-1])

    ecg_filenames = np.asarray(ecg_filenames)

    # Gender processing - replace with nicer code later
    gender = np.asarray(gender)
    gender[np.where(gender == "Male")] = 0
    gender[np.where(gender == "male")] = 0
    gender[np.where(gender == "M")] = 0
    gender[np.where(gender == "Female")] = 1
    gender[np.where(gender == "female")] = 1
    gender[np.where(gender == "F")] = 1
    gender[np.where(gender == "NaN")] = 2
    gender = gender.astype(np.int)

    # Age processing - replace with nicer code later
    age = np.asarray(age)
    age[np.where(age == "NaN")] = -1
    age = age.astype(np.int)

    # Load SNOMED codes
    SNOMED_scored=pd.read_csv("SNOMED_mappings_scored.csv", sep=";")
    SNOMED_unscored=pd.read_csv("SNOMED_mappings_unscored.csv", sep=";")

    # Load labels to dataframe
    df_labels = pd.DataFrame(labels)

    # Remove unscored labels
    for i in range(len(SNOMED_unscored.iloc[0:,1])):
        df_labels.replace(to_replace=str(SNOMED_unscored.iloc[i,1]), inplace=True ,value="undefined class", regex=True)

    # Replace overlaping SNOMED codes
    '''
    codes_to_replace=['713427006','284470004','427172004']
    replace_with = ['59118001','63593006','17338001']

    for i in range(len(codes_to_replace)):
        df_labels.replace(to_replace=codes_to_replace[i], inplace=True ,value=replace_with[i], regex=True)
    '''
    # One-Hot encode classes
    one_hot = MultiLabelBinarizer()
    y=one_hot.fit_transform(df_labels[0].str.split(pat=','))
    y= np.delete(y, -1, axis=1)
    classes_for_prediction = one_hot.classes_[0:-1]

    global order_array
    order_array = np.arange(0,y.shape[0],1)

    print(classes_for_prediction)
    print("classes: {}".format(y.shape[1]))

    # Train model.
    print('Training model...')


    model=create_model()
    batchsize = 30
    class_dict= class_dict={0: 63.62172285, 1: 12.54579025, 2: 5.42889102, 3: 60.23758865, 4: 18.14850427, 5: 18.54475983,
    6: 4.03971463, 7: 55.33224756 , 8: 52.26769231, 9: 34.04208417, 10: 7.90828678, 11: 10.91709512, 12: 3.1032152, 13: 7.97886332, 
    14: 66.09727626 , 15: 0.90529738,16: 7.86071263, 17: 100.5147929, 18: 15.11298932, 19: 10.49227918, 20: 43.78092784, 
    21: 7.87528975, 22: 16.91932271, 23: 88.93717277 , 24: 18.81173865, 25: 11.6829436 , 26: 27.75653595 }

    #model.fit_generator(generator=batch_generator(batch_size=batchsize, gen_x=generate_X(input_directory), gen_y=generate_y(y), 
    #gen_z=generate_z(age,gender), ohe_labels = classes_for_prediction),steps_per_epoch=(len(y)/batchsize), epochs=1)
    
    #HUSK Å LEGGE TIL CLASS_DICT

    model.fit(x=batch_generator(batch_size=batchsize, gen_x=generate_X(ecg_filenames), gen_y=generate_y(y),gen_z=generate_z(age,gender), ohe_labels=classes_for_prediction), 
    epochs=50, steps_per_epoch=(len(y)/batchsize), class_weight=class_dict, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='AUC', factor=0.1, patience=1, verbose=1, mode='max',
min_delta=0.0001, cooldown=0, min_lr=0),tf.keras.callbacks.EarlyStopping(monitor='AUC', mode='max', verbose=1, patience=2)])

    # Save model.
    print('Saving model...')
    #model.save("model.h5")
    filename = os.path.join(output_directory, 'model.h5')
    model.save_weights(filename)

    #final_model={'model':model, 'imputer':imputer,'classes':classes}

    #filename = os.path.join(output_directory, 'finalized_model.sav')
    #joblib.dump(final_model, filename, protocol=0)

# Load challenge data.
def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data

# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)

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


def generate_y(y):
    while True:
        for i in order_array:
            y_train = y[i]
            yield y_train

def generate_X(ecg_filenames):
    while True:
        for i in order_array:
            data, header_data = load_challenge_data(ecg_filenames[i])
            X_train_new = keras.preprocessing.sequence.pad_sequences(data, maxlen=5000, truncating='post',padding="post")
            X_train_new = X_train_new.reshape(5000,12)
            yield X_train_new

def generate_z(age, gender):
    while True:
        for i in order_array:
            gen_age = age[i]
            gen_gender = gender[i]
            z_train = [gen_age , gen_gender]
            yield z_train

def batch_generator(batch_size, gen_x,gen_y, gen_z, ohe_labels):
    np.random.shuffle(order_array)
    batch_features = np.zeros((batch_size,5000, 12))
    batch_labels = np.zeros((batch_size,len(ohe_labels)))
    batch_demo_data = np.zeros((batch_size,2))
    while True:
        for i in range(batch_size):

            batch_features[i] = next(gen_x)
            batch_labels[i] = next(gen_y)
            batch_demo_data[i] = next(gen_z)

        X_combined = [batch_features, batch_demo_data]
        yield X_combined, batch_labels

