# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:16:50 2024

@author: MANODEEP RAY
"""

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
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
from sklearn.metrics import accuracy_score
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from sklearn.metrics import f1_score, confusion_matrix, classification_report



def plotting():
    
    return


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

def load_model():
    
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

    return model

def preprocess(train_path , test_path):
    
    training = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    training = training.sample(frac=1, random_state=42) # randomizing the filenames since the files are graded in alphabetical order
    test = test.sample(frac=1, random_state=42)
    x = training['fname'].values
    y = training['label'].values
        
    X = []  # Store MFCC features
    Y = []  # Store class labels

    
    
    for index, row in training.iterrows():
            audio_path = row['fname']
            class_label = row['label']
            audio_path = 'wavfiles/' + audio_path
            mfccs = preprocess_audio(audio_path)
            print(len(mfccs))
            mfccs = pad_or_truncate_mfccs(mfccs, target_length=200)  # Adjust target_length as needed
            X.append(mfccs)
            Y.append(class_label)
        
    
    X = np.array(X)
    Y = np.array(Y)
    class_mapping = {'grade1': 0, 'grade2': 1, 'grade3': 2 ,'grade4': 3 }
    Y = [class_mapping[label] for label in Y]
    
    Y = to_categorical(Y)  # One-hot encode labels
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    return X , Y , X_train, X_val, y_train, y_val

def evaluate(y_true,y_pred , history):
    
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


def prediction(directory_path):
        
    
    # Initialize an empty DataFrame
    train = pd.DataFrame(columns=['File', 'Predicted_Class'])
    
    for file in tqdm(os.listdir(directory_path)):
            # Predict the class
            audio_path = os.path.join(directory_path, file)
            mfccs_test = preprocess_audio(audio_path)  # Assuming you have a function preprocess_audio
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
    predicted_train = pd.read_csv('traing_audio_validation.csv')
    
    #re-initializing the dataframes
    training = pd.read_csv('training.csv')
    test = pd.read_csv('test.csv')
    
    
    y_predicted_train = predicted_train.iloc[:,-1].values
    y_true_train = training.iloc[:,-1].values
    y_predicted_test = predicted_test.iloc[:,-1].values
    y_true_test = test.iloc[:,-1].values
    
    classes = list(np.unique(training.label))
    
    labels = np.array(classes)
    y_test = classes
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    y_test_onehot = to_categorical(y_test_encoded, num_classes=len(label_encoder.classes_))
    
    
    y_pred = predicted_train['Predicted_Class'].values
    y_test_encoded = label_encoder.fit_transform(y_pred)
    y_pred = y_test_encoded
    
    y_true = training['label'].values
    y_true_encoded = label_encoder.fit_transform(y_true)
    y_true = y_true_encoded

    return y_true  , y_pred



def predict_single(file , model):
        
    def downsample_audio(audio_path, target_sr=22050):
        y, sr = librosa.load(audio_path)
        return librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    def extract_mfcc_features(audio, sr, n_mfcc=20):
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return mfccs.T
    def pad_or_truncate_mfccs(mfccs, target_length):
        if len(mfccs) > target_length:
            return mfccs[:target_length, :]
        else:
            return np.pad(mfccs, ((0, target_length - len(mfccs)), (0, 0)), mode='constant')
    def preprocess_audio(audio_path, target_sr=22050, n_mfcc=20):
        audio, sr = librosa.load(audio_path, sr=target_sr)  # Load audio and sample rate
        audio = downsample_audio(audio_path, target_sr)
        sr = target_sr
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20).T
        mfccs = pad_or_truncate_mfccs(mfccs, target_length=200)  # Adjust target_length to match the expected input shape
        return mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)  # Reshape for prediction
      # Reshape for prediction
    
    
    
    
    # Predict the class
    audio_path = 'test/'+file+'.wav'  # Replace with the actual path
    mfccs_test = preprocess_audio(audio_path)
    
    mfccs_test.shape
    
    prediction = model.predict(mfccs_test)
    predicted_class = np.argmax(prediction)  # Get the index of the predicted class
    
    # Map the index back to the class label (if applicable)
    class_mapping = {0: 'grade1', 1: 'grade2', 2: 'grade3', 3: 'grade4'}  # Adjust accordingly
    predicted_label = class_mapping[predicted_class]
    
    print("Predicted class:", predicted_class)
    print("Predicted label:", predicted_label)


    return True

def predict_single(file, model):
    
    
    
    def downsample_audio(audio_path, target_sr=22050):
        y, sr = librosa.load(audio_path)
        return librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    def extract_mfcc_features(audio, sr, n_mfcc=20):
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return mfccs.T
    def pad_or_truncate_mfccs(mfccs, target_length):
        if len(mfccs) > target_length:
            return mfccs[:target_length, :]
        else:
            return np.pad(mfccs, ((0, target_length - len(mfccs)), (0, 0)), mode='constant')
    def preprocess_audio(audio_path, target_sr=22050, n_mfcc=20):
        audio, sr = librosa.load(audio_path, sr=target_sr)  # Load audio and sample rate
        audio = downsample_audio(audio_path, target_sr)
        sr = target_sr
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20).T
        mfccs = pad_or_truncate_mfccs(mfccs, target_length=200)  # Adjust target_length to match the expected input shape
        return mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)  # Reshape for prediction
      # Reshape for prediction
    
    
    
    
    # Predict the class
    audio_path = 'test/'+file+'.wav'  # Replace with the actual path
    mfccs_test = preprocess_audio(audio_path)
    
    mfccs_test.shape
    
    prediction = model.predict(mfccs_test)
    predicted_class = np.argmax(prediction)  # Get the index of the predicted class
    
    # Map the index back to the class label (if applicable)
    class_mapping = {0: 'grade1', 1: 'grade2', 2: 'grade3', 3: 'grade4'}  # Adjust accordingly
    predicted_label = class_mapping[predicted_class]
    
    print("Predicted class:", predicted_class)
    print("Predicted label:", predicted_label)
        


if __name__ == "__main__":
    train_path , test_path = 'training.csv' , 'test.csv'
    X , Y , X_train, X_val, y_train, y_val = preprocess(train_path , test_path)
    model = load_model()
    model.compile(loss='categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val))
    directory_path = 'wavfiles'
    y_true  , y_pred = prediction(directory_path)
    
    
    
    evaluate(y_true,y_pred , history)
    
    
    file = str(input("enter the file name :"))
    predict_single(file, model)
    
    
    
    
