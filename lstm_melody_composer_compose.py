# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:08:38 2015

@author: Konstantin
"""

import data_utils_compose
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.recurrent import LSTM
import numpy as np
import glob
from os import listdir

np.set_printoptions(threshold=np.nan) #Comment that line out, to print reduced matrices


#User Info
print()
print("User Information:")
print("This is a tool for composing melodies to given chord sequences with a LSTM Recurrent Neural Network that has already been trained.")
print("It has been created in Fall 2015 by Konstantin Lackner under the supervision of Thomas Volk and Prof. Diepold at the Chair of Data Processing at the Technical University of Munich (TUM).")
#print("For more information please visist: ...")
print()


chord_dir = './testData/chords/'
composition_dir = './testData/melody_composition/'

print("Put the chord sequences in MIDI format in the directory: %s. Nothing else but the MIDI files should be in that directory!" %(chord_dir))
print("The composed melodies will be stored in MIDI format in the directory: %s" %(composition_dir))
print("Chord notes must be between C2 and B2.")
print()
print()
print("LSTM RNN Composer:")
print()


chord_files = glob.glob("%s*.mid" %(chord_dir))

composition_files = []
for i in range(len(chord_files)):
    composition_files.append('%d' %(i+1))

mel_lowest_note = 60

print()
print("Using the following files as Test Chords:")
print(chord_files)
print()
print("Melodies will be saved to the files:")
print(composition_files)
print()

print("Choose a resolution factor. (e.g. Resolution_Factor=24: 1/8 Resolution, 12: 1/16 Resolution, 6: 1/32 Resolution, etc...)")
resolution_factor = int(input('Resolution Factor (recommended=12):')) #24: 1/8 Resolution, 12: 1/16 Resolution, 6: 1/32 Resolution


chord_lowest_note, chord_highest_note, chord_ticks = data_utils_compose.getNoteRangeAndTicks(chord_files, res_factor=resolution_factor)

chord_roll = data_utils_compose.fromMidiCreatePianoRoll(chord_files, chord_ticks, chord_lowest_note,
                                                        res_factor=resolution_factor)

double_chord_roll = data_utils_compose.doubleRoll(chord_roll)

test_data = data_utils_compose.createNetInputs(double_chord_roll, seq_length=chord_ticks)

batch_size = 128
class_mode = "binary"

print()
print()
print("Enter the threshold (threshold is used for creating a Piano Roll Matrix out of the Network Output)")
thresh = float(input('Threshold (recommended ~ 0.1):'))

print()
print()
print("Loading Model and Weights...")
print()

#Load model file
model_dir = './saved_model/'
model_files = listdir(model_dir)
print("Choose a file for the model:")
print("---------------------------------------")
for i, file in enumerate(model_files):
    print(str(i) + " : " + file)
print("---------------------------------------")
print()
file_number_model = int(input('Type in the number in front of the file you want to choose:')) 
model_file = model_files[file_number_model]
model_path = '%s%s' %(model_dir, model_file)

#Load weights file
weights_dir = './weights/'
weights_files = listdir(weights_dir)
print()
print()
print("Choose a file for the weights (Model and Weights MUST correspond!):")
print("---------------------------------------")
for i, file in enumerate(weights_files):
    print(str(i) + " : " + file)
print("---------------------------------------")
print()
file_number_weights = int(input('Type in the number in front of the file you want to choose:')) 
weights_file = weights_files[file_number_weights]
weights_path = '%s%s' %(weights_dir, weights_file)


print()
print("loading model...")
model = model_from_json(open(model_path).read())
print()
print("loading weights...")
model.load_weights(weights_path)
print()
print("Compiling model...")
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode=class_mode)

print()
print("Compose...")
for i, song in enumerate(test_data):
    net_output = model.predict(song)
    #print("net_output:", net_output)
    net_roll = data_utils_compose.NetOutToPianoRoll(net_output, threshold=thresh)
    #print("net_roll:", net_roll)
    #print("net_roll.shape", net_roll.shape)
    data_utils_compose.createMidiFromPianoRoll(net_roll, mel_lowest_note, composition_dir,
                                               composition_files[i], thresh, res_factor=resolution_factor)
    
    print("Finished composing song %d." %(i+1))

print()    
print("Dope!")


