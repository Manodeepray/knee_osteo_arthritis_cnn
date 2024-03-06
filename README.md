# knee_osteo_arthritis_cnn
this  is a cnn model for classifying  the level/ grade of knee osteo arthritis a patient has using audio signals from VGA machine .
i have extracted mfccs from each audio signal as feature for this model usnf "python speech feature" library 
before doing so i have named all the files as the grade they are and assigned them a number for example "Grade1 (22).wav"
each aidio signal is read using librosa ad then they are downsampled , mfccs extracted and then truncated  to a fixed length i.e 200 here

you can find the name of the audio signals  in "training.csv" and for testing "test.csv"
the audio files are stored in "wavefiles" folder and the "test" folder respectively

the "train_audio_validation.csv" and "test_audio_validation.csv" are later used for generating the confusion matrix for the model

you can load the model with the pretrained weights from "model_1/conv2.h" directory

STATS:
 max accuracy : 100% (slight overfitting due to low amount of data )
 max validation accuracy : 82% 
