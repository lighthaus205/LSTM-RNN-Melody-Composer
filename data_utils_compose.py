# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:07:48 2015

@author: Konstantin
"""

from mido import MidiFile, MidiTrack, Message
from mido.midifiles_meta import MetaMessage
import numpy as np
#np.set_printoptions(threshold=np.nan)


def getNoteRangeAndTicks(files_dir, res_factor=1):
    ticks = []
    notes = []
    for file_dir in files_dir:
        file_path = "%s" %(file_dir)
        mid = MidiFile(file_path)                   
        
        for track in mid.tracks: #preprocessing: Checking range of notes and total number of ticks
            num_ticks = 0           
            for message in track:
                if not isinstance(message, MetaMessage):
                    notes.append(message.note)
                    num_ticks += int(message.time/res_factor)
            ticks.append(num_ticks)
                    
    return min(notes), max(notes), max(ticks)
    

def fromMidiCreatePianoRoll(files_dir, ticks, lowest_note, res_factor=1):
    num_files = len(files_dir)        
        
    piano_roll = np.zeros((num_files, ticks, 12))
    
    for i, file_dir in enumerate(files_dir):
        file_path = "%s" %(file_dir)
        mid = MidiFile(file_path)
        note_time_onoff = getNoteTimeOnOffArray(mid, res_factor)
        note_on_length = getNoteOnLengthArray(note_time_onoff)
        for message in note_on_length:
            piano_roll[i, message[1]:(message[1]+int(message[2]/2)), message[0]-lowest_note] = 1
    
    return piano_roll


def getNoteTimeOnOffArray(mid, res_factor):
    
    note_time_onoff_array = []  
    
    for track in mid.tracks:
        current_time = 0
        for message in track:
            if not isinstance(message, MetaMessage):
                current_time += int(message.time/res_factor)
                if message.type == 'note_on':
                    note_onoff = 1
                elif message.type == 'note_off':
                    note_onoff = 0
                else:
                    print("Error: Note Type not recognized!")
                    
                note_time_onoff_array.append([message.note, current_time, note_onoff])
                
    return note_time_onoff_array
    
    
def getNoteOnLengthArray(note_time_onoff_array):
    note_on_length_array = []
    for i, message in enumerate(note_time_onoff_array):
        if message[2] == 1: #if note type is 'note_on'
            start_time = message[1]
            for event in note_time_onoff_array[i:]: #go through array and look for, when the current note is getting turned off
                if event[0] == message[0] and event[2] == 0:
                    length = event[1] - start_time
                    break
                
            note_on_length_array.append([message[0], start_time, length])
            
    return note_on_length_array
    
    
def doubleRoll(roll):
    double_roll = []
    for song in roll:
        double_song = np.zeros((roll.shape[1]*2, roll.shape[2]))
        double_song[0:roll.shape[1], :] = song
        double_song[roll.shape[1]:, :] = song
        double_roll.append(double_song)
        
    return np.array(double_roll)


def createNetInputs(roll, seq_length=3072):
    #roll: 3-dim array with Midi Files as piano roll. Size: (num_samples=num Midi Files, num_timesteps, num_notes)
    #seq_length: Sequence Length. Length of previous played notes in regard of the current note that is being trained on
    #seq_length in Midi Ticks. Default is 96 ticks per beat --> 3072 ticks = 8 Bars
    
    testData = []
    
    
    for song in roll:
        pos = 0
        X = []
        while pos+seq_length < song.shape[0]:
            sequence = np.array(song[pos:pos+seq_length])
            X.append(sequence)
            pos += 1
            
        testData.append(np.array(X))

    
    return np.array(testData)



def NetOutToPianoRoll(network_output, threshold=0.1):
    piano_roll = []
    for i, timestep in enumerate(network_output):
        if np.amax(timestep) > threshold:
            pos = 0
            pos = np.argmax(timestep)
            timestep[:] = np.zeros(timestep.shape)
            timestep[pos] = 1
        else:
            timestep[:] = np.zeros(timestep.shape)
        piano_roll.append(timestep)
        
    return np.array(piano_roll)


def createMidiFromPianoRoll(piano_roll, lowest_note, directory, mel_test_file, threshold, res_factor=1):
    
    ticks_per_beat = int(96/res_factor)
    mid = MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    mid_files = []
    

    delta_times = [0]
    for k in range(piano_roll.shape[1]):#initial starting values
        if piano_roll[0, k] == 1:
            track.append(Message('note_on', note=k+lowest_note, velocity=100, time=0))
            delta_times.append(0)
        
    for j in range(piano_roll.shape[0]-1):#all values between first and last one
        set_note = 0 #Check, if for the current timestep a note has already been changed (set to note_on or note_off)
        
        for k in range(piano_roll.shape[1]):
            if (piano_roll[j+1, k] == 1 and piano_roll[j, k] == 0) or (piano_roll[j+1, k] == 0 and piano_roll[j, k] == 1):#only do something if note_on or note_off are to be set
                if set_note == 0:
                    time = j+1 - sum(delta_times)          
                    delta_times.append(time)
                else:
                    time = 0
                    
                if piano_roll[j+1, k] == 1 and piano_roll[j, k] == 0:
                    set_note += 1
                    track.append(Message('note_on', note=k+lowest_note, velocity=100, time=time))
                if piano_roll[j+1, k] == 0 and piano_roll[j, k] == 1:
                    set_note += 1
                    track.append(Message('note_off', note=k+lowest_note, velocity=64, time=time))
                           
    mid.save('%s%s_th%s.mid' %(directory, mel_test_file, threshold))
    mid_files.append('%s.mid' %(mel_test_file))
       
    return


    
