# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:08:18 2015

@author: Konstantin
"""

import data_utils_train
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
import numpy as np
import time
import csv
import glob

np.set_printoptions(threshold=np.nan) #Comment that line out, to print reduced matrices


#User Info
print()
print("User Information:")
print("This is a tool for training a LSTM Recurrent Neural Network to learn melodies to given chord sequences.")
print("It has been created in Fall 2015 by Konstantin Lackner under the supervision of Thomas Volk and Prof. Diepold at the Chair of Data Processing at the Technical University of Munich (TUM).")
#print("For more information please visist: ...")
print()


chord_train_dir = './trainData/chords/'
mel_train_dir = './trainData/melody/'

print("Put the chords in MIDI format in the directory: %s" %(chord_train_dir))
print("Put the melodies in MIDI format in the directory: %s" %(mel_train_dir))
print("ALL MIDI files need to be of the same length (e.g. 8 bars).")
print("Chord notes must be between C2 and B2. Melody notes must be between C3 and B4.")
print("Keep the chord and CORRESPONDING melody files in the SAME ORDER within their respective folders.")
print()
print()
print("LSTM RNN Trainer:")
print()
chord_train_files = glob.glob("%s*.mid" %(chord_train_dir))
mel_train_files = glob.glob("%s*.mid" %(mel_train_dir))



print("Choose a resolution factor. (e.g. Resolution_Factor=24: 1/8 Resolution, 12: 1/16 Resolution, 6: 1/32 Resolution, etc...)")
resolution_factor = int(input('Resolution Factor (recommended=12):')) #24: 1/8 Resolution, 12: 1/16 Resolution, 6: 1/32 Resolution

#Preprocessing: Get highest and lowest notes + maximum midi_ticks overall midi files
chord_lowest_note, chord_highest_note, chord_ticks = data_utils_train.getNoteRangeAndTicks(chord_train_files, res_factor=resolution_factor)
mel_lowest_note, mel_highest_note, mel_ticks = data_utils_train.getNoteRangeAndTicks(mel_train_files, res_factor=resolution_factor)

#Create Piano Roll Representation of the MIDI files. Return: 3-dimensional array or shape (num_midi_files, maximum num of ticks, note range)
chord_roll = data_utils_train.fromMidiCreatePianoRoll(chord_train_files, chord_ticks, chord_lowest_note, chord_highest_note,
                                                res_factor=resolution_factor)
mel_roll = data_utils_train.fromMidiCreatePianoRoll(mel_train_files, mel_ticks, mel_lowest_note, mel_highest_note,
                                              res_factor=resolution_factor)



#Double each chord_roll and mel_roll. Preprocessing to create Input and Target Vector for Network
double_chord_roll = data_utils_train.doubleRoll(chord_roll)
double_mel_roll = data_utils_train.doubleRoll(mel_roll)

#Create Network Inputs:
#Input_data Shape: (num of training samples, num of timesteps=sequence length, note range)
#Target_data Shape: (num of training samples, note range)
input_data, target_data = data_utils_train.createNetInputs(double_chord_roll, double_mel_roll, seq_length=chord_ticks)
input_data = input_data.astype(np.bool)
target_data = target_data.astype(np.bool)


input_dim = input_data.shape[2]
output_dim = target_data.shape[1]


print()
print("For how many epochs do you wanna train?")
num_epochs = int(input('Num of Epochs:'))
print()

print()
print("Choose a batch size:")
print("(Batch size determines how many training samples per gradient-update are used. --> Number of gradient-updates per epoch: Num of samples / batch size)")
batch_size = int(input('Batch Size (recommended=128):'))
print()

print()
print("Network Input Dimension:", input_dim)
print("Network Output Dimension:", output_dim)
print("How many layers should the network have?")
num_layers = int(input('Number of Layers:'))
print()




#Building the Network
model = Sequential()
if num_layers == 1:
    print("Your Network:")
    model.add(LSTM(input_dim=input_dim, output_dim=output_dim, activation='sigmoid', return_sequences=False))
    print("add(LSTM(input_dim=%d, output_dim=%d, activation='sigmoid', return_sequences=False))" %(input_dim, output_dim))
elif num_layers > 1:
    print("Enter the number of units for each layer:")
    num_units = []
    for i in range(num_layers-1):
        units = int(input('Number of Units in Layer %d:' %(i+1)))
        num_units.append(units)
    print()
    print("Your Network:")
    model.add(LSTM(input_dim=input_dim, output_dim=num_units[0], activation='sigmoid', return_sequences=True))
    print("add(LSTM(input_dim=%d, output_dim=%d, activation='sigmoid', return_sequences=True))" %(input_dim, num_units[0]))
    for i in range(num_layers-2):
        model.add(LSTM(output_dim=num_units[i+1], activation='sigmoid', return_sequences=True))
        print("add(LSTM(output_dim=%d, activation='sigmoid', return_sequences=True))" %(num_units[i+1]))
    model.add(LSTM(output_dim=output_dim, activation='sigmoid', return_sequences=False))
    print("add(LSTM(output_dim=%d, activation='sigmoid', return_sequences=False))" %(output_dim))


print()
print()
print("Compiling your network with the following properties:")
loss_function = 'binary_crossentropy'
optimizer = 'adam'
class_mode = 'binary'
print("Loss function: ", loss_function)
print("Optimizer: ", optimizer)
print("Class Mode: ", class_mode)
print("Number of Epochs: ", num_epochs)
print("Batch Size: ", batch_size)

model.compile(loss=loss_function, optimizer=optimizer, class_mode=class_mode)


print()
print("Training...")
history = data_utils_train.LossHistory()
model.fit(input_data, target_data, batch_size=batch_size, nb_epoch=num_epochs, callbacks=[history])
w = csv.writer(open("./history_csv/%dlayer_%sepochs_%s.csv" %(num_layers, num_epochs, time.strftime("%Y%m%d_%H_%M")), "w"))
for loss in history.losses:
    w.writerow([loss])


print()
print("Saving model and weights...")
print("Saving weights...")
weights_dir = './weights/'
weights_file = '%dlayer_%sepochs_%s' %(num_layers, num_epochs, time.strftime("%Y%m%d_%H_%M.h5"))
weights_path = '%s%s' %(weights_dir, weights_file)
print("Weights Path:", weights_path)
model.save_weights(weights_path)

print("Saving model...")
json_string = model.to_json()
model_file = '%dlayer_%sepochs_%s' %(num_layers, num_epochs, time.strftime("%Y%m%d_%H_%M.json"))
model_dir = './saved_model/'
model_path = '%s%s' %(model_dir, model_file)
print("Model Path:", model_path)
open(model_path, 'w').write(json_string)

print()
print("Dope!")