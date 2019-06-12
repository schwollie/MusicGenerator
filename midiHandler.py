import os
from PianoRoll import MidiFile
import pretty_midi as pm
import matplotlib.pyplot as im
import numpy as np
from mido import MidiFile, MidiTrack, Message as MidiMessage


def cycle_folders(dir="midi"):
    music = []
    count = -1
    for root, dirs, files in os.walk(dir):
        for name in files:
            music.append([os.path.join(root, name), count])
            #print(name)
        count += 1
    # 0 : albeniz; 1 : beethoven; 2 : chopin; 3 : mozart;
    return music


def create_midi_rolls(music):
    for m in range(len(music)):  # music = list [[ path , author ]]
        mid = MidiFile(music[m][0])
        r = mid.get_roll()[0]  # [0] because we only need first channel
        mid.draw_roll()
        print(r)
        break


def test():  # working !!
    music = cycle_folders()
    for m in range(len(music)):
        mid = music[m][0]
        mid = pm.PrettyMIDI(mid)
        s = pm.PrettyMIDI.get_piano_roll(mid, fs=32)  # 32 showed best results up to now
        print(s)
        print(s.shape)
        #im.imshow(s)
        #im.show()

        new_mid = piano_roll_to_midi(s)
        new_mid.save("output/output.mid")

        if m==1: return


def piano_roll_to_midi(piano_roll, base_note=0):
    """Convert piano roll to a MIDI file."""
    notes, frames = piano_roll.shape
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    now = 0
    piano_roll = np.hstack((np.zeros((notes, 1)),
                            piano_roll,
                            np.zeros((notes, 1))))
    velocity_changes = np.nonzero(np.diff(piano_roll).T)
    for time, note in zip(*velocity_changes):
        velocity = piano_roll[note, time + 1]
        message = MidiMessage(
            type='note_on' if velocity > 0 else 'note_off',
            note=min(int(note + base_note), 127),
            velocity=min(int(velocity * 127), 127),
            time=int((time - now)*20))
        track.append(message)
        now = time
    return midi


if __name__ == "__main__" :
    #m = cycle_folders()
    #create_midi_rolls(m)
    test()

