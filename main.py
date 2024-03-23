# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 12:44:19 2024

@author: MANODEEP
"""
print("importing libraries")
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa

from tensorflow.keras.layers import (
        Conv2D,
        MaxPool2D,
        Flatten,
        LSTM,
        TimeDistributed,
        LayerNormalization,
        Dense,
        Dropout,
    )
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import accuracy_score
import pickle
import tensorflow as tf

from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report





def downsample_audio(audio_path, target_sr=22050):
    y, sr = librosa.load(audio_path)
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr)

def extract_mfcc_features(audio, sr, n_mfcc=20):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T

def preprocess_audio(audio_path, target_sr=22050, n_mfcc=20):
    audio = downsample_audio(audio_path, target_sr)
    sr = target_sr
    mfccs = extract_mfcc_features(audio, sr, n_mfcc)
    return mfccs.reshape(mfccs.shape[0], mfccs.shape[1], 1)

def pad_or_truncate_mfccs(mfccs, target_length):
    if len(mfccs) > target_length:
        return mfccs[:target_length, :]
    else:
        return np.pad(mfccs, ((0, target_length - len(mfccs)), (0, 0)), mode='constant')
    

def complete_preprocessing(training,test):
#re-fitting the model for using .history 

    from sklearn.model_selection import train_test_split

    training = training.sample(frac=1, random_state=42)  #shuffeling
    test = test.sample(frac=1, random_state=42)
    x = training['fname'].values
    y = training['label'].values

    x_train, x_test , y_train , y_test = train_test_split(x , y ,test_size = 0.2 ,random_state  =  0)
            
    classes = list(np.unique(training.label))
    
    
    X = []  # Store MFCC features
    Y = []  # Store class labels
    
    
    for index, row in training.iterrows():
            audio_path = row['fname']
            class_label = row['label']
            audio_path = 'wavfiles/' + audio_path
            mfccs = preprocess_audio(audio_path)
            mfccs = pad_or_truncate_mfccs(mfccs, target_length=200)  # Adjust target_length as needed
            X.append(mfccs)
            Y.append(class_label)
        
    
    X = np.array(X)
    Y = np.array(Y)
    
    
    #convering grade to numbers 0 1 2 3 
    class_mapping = {'grade1': 0, 'grade2': 1, 'grade3': 2 ,'grade4': 3 }
    Y = [class_mapping[label] for label in Y]
    
    Y = to_categorical(Y)  # One-hot encode labels
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, X_val,x_test, y_train, y_val , y_test , mfccs



#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Dropout(0.25))

def load_model(mfccs):
    
    num_classes = 4
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(mfccs.shape[0], mfccs.shape[1], 1)))
    
    
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu' , kernel_regularizer=regularizers.l2(0.01)))
    
    
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.summary()
    # Output layer for 4 classes

    return model


def preprocess_audio_prediction(audio_path, target_sr=22050, n_mfcc=20):
    audio, sr = librosa.load(audio_path, sr=target_sr)  # Load audio and sample rate
    audio = downsample_audio(audio_path, target_sr)
    sr = target_sr
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20).T
    mfccs = pad_or_truncate_mfccs(mfccs, target_length=200)  # Adjust target_length to match the expected input shape
    return mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)  # Reshape for prediction
  # Reshape for prediction


def prediction_preprocess(training,directory_path_1,directory_path_2):  
    classes = list(np.unique(training.label))
    
    
    X = []  # Store MFCC features
    Y = []  # Store class labels
    
    
    #making prediction of the signals in "test"
    
    test = pd.DataFrame(columns=['File', 'Predicted_Class'])

    for file in tqdm(os.listdir(directory_path_1)):
        # Predict the class
        audio_path = 'test/'+file  # Replace with the actual path
        mfccs_test = preprocess_audio_prediction(audio_path)
        prediction = model.predict(mfccs_test)
        predicted_class = np.argmax(prediction)  # Get the index of the predicted class
    
        # Map the index back to the class label (if applicable)
        class_mapping = {0: 'grade1', 1: 'grade2', 2: 'grade3', 3: 'grade4'}  # Adjust accordingly
        predicted_label = class_mapping[predicted_class]
    
        print("Predicted class for "+file +":", predicted_class)
        print("Predicted label"+file +":", predicted_label)
        test = test.append({'File': file, 'Predicted_Class': predicted_label}, ignore_index=True)

        
    #making prediction of the signals in "wavfile" 
        
    
    
    # Initialize an empty DataFrame
    train = pd.DataFrame(columns=['File', 'Predicted_Class'])

    for file in tqdm(os.listdir(directory_path_2)):
            # Predict the class
            audio_path = os.path.join(directory_path_2, file)
            mfccs_test = preprocess_audio_prediction(audio_path)  # Assuming you have a function preprocess_audio
            prediction = model.predict(mfccs_test)
            predicted_class = np.argmax(prediction)

            # Map the index back to the class label (if applicable)
            class_mapping = {0: 'grade1', 1: 'grade2', 2: 'grade3', 3: 'grade4'}  # Adjust accordingly
            predicted_label = class_mapping[predicted_class]

            # Append the result to the DataFrame
            train = train.append({'File': file, 'Predicted_Class': predicted_label}, ignore_index=True)


    test = test.to_csv('test_audio_validation.csv', index =False)
    train = train.to_csv('train_audio_validation.csv', index =False)


    predicted_test = pd.read_csv('test_audio_validation.csv')
    predicted_train = pd.read_csv('train_audio_validation.csv')

     
    #re-initializing the dataframes
    training = pd.read_csv('training.csv')
    test = pd.read_csv('test.csv')
    
    
    y_predicted_train = predicted_train.iloc[:,-1].values
    y_true_train = training.iloc[:,-1].values
    y_predicted_test = predicted_test.iloc[:,-1].values
    y_true_test = test.iloc[:,-1].values
    
    return classes , y_true_test ,y_predicted_test , y_true_train , y_predicted_train , predicted_test ,predicted_train


def plot_graph(history,classes ,training , test , predicted_train):
   
   
    
   #plotting the accuracy 
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    
    #plotting the loss graph
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    
    
    labels = np.array(classes)
    
    
    
    # Assuming y_test contains string labels
    y_test = classes
    
    # Use LabelEncoder to convert string labels to integers
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    
    # One-hot encode y_test_encoded
    y_test_onehot = to_categorical(y_test_encoded, num_classes=len(label_encoder.classes_))
    
    # Display the original and one-hot encoded labels
    print("Original labels:", y_test)
    print("Encoded labels:", y_test_encoded)
    print("One-hot encoded labels:")
    print(y_test_onehot)
    
    
    
    y_pred = predicted_train['Predicted_Class'].values
    y_test_encoded = label_encoder.fit_transform(y_pred)
    y_pred = y_test_encoded
    
    y_true = training['label'].values
    y_true_encoded = label_encoder.fit_transform(y_true)
    y_true = y_true_encoded
    
    
    #confusion matrix
    
    
    # Calculate F1 score
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Print classification report
    print("Classification Report:\n", classification_report(y_true, y_pred))
    
    # Print F1 score
    print("Weighted F1 Score:", f1)
    
    
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    
    # Find the maximum accuracy achieved during training
    max_train_accuracy = max(train_accuracy)
    max_val_accuracy = max(val_accuracy)
    
    print(f"Maximum Training Accuracy: {max_train_accuracy}")
    print(f"Maximum Validation Accuracy: {max_val_accuracy}")

    return True

def callback():
    checkpoint_callback = ModelCheckpoint(filepath='weight/best_weights.keras',
                                  monitor='val_accuracy',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='max')
    return checkpoint_callback

if __name__ =="__main__":
    print("importing libraries....")
    
    
    print("loading dataset....")
    training = pd.read_csv('training.csv')
    test = pd.read_csv('test.csv')
    
    
    print("initial preprocessing data....")
    X_train, X_val,x_test, y_train, y_val , y_test,mfccs = complete_preprocessing(training,test)
    
    
    print("loading model....")
    model = load_model(mfccs)
    print("model loaded....")

    
    
        
    
    last_layer_output_shape = model.layers[-1].output_shape
    print("Last layer output shape:", last_layer_output_shape)
    
    print("compiling....")

    model.compile(loss='categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    
    print("training model....")

    checkpoint_callback = callback()
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val),callbacks=[checkpoint_callback])
    print("model trained....")



    print("reloading weights....")

    
    model.load_weights("weight/best_weights.keras")
    
    directory_path_1 = 'test'
    directory_path_2 = 'wavfiles'
    
    
    print("preprocessing data for prediction....")

    classes , y_true_test ,y_predicted_test , y_true_train , y_predicted_train , predicted_test ,predicted_train = prediction_preprocess(training,directory_path_1,directory_path_2)
    
    
    print("re initializing dataset....")

    training = pd.read_csv('training.csv')
    test = pd.read_csv('test.csv') 
    
    print("plotting graph....")

    plot_graph(history,classes ,training , test , predicted_train)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    