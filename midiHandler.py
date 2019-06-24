import os
import pickle

import numpy as np
import pretty_midi as pm
from mido import MidiFile, MidiTrack, Message as MidiMessage
from sklearn.decomposition import TruncatedSVD


def print_progress(total, now):
    print(str(now / total * 100) + " %")


def cycle_folders(dir="midi"):
    music = []
    count = 0
    for root, dirs, files in os.walk(dir, topdown=True):
        for name in sorted(files):
            music.append([os.path.join(root, name), count])
            # print(name)
        count += 1
    # 1 : albeniz; 2 : beethoven; 3 : chopin; 4 : mozart; (on windows os does not walk in alphabetic order)
    # 0 is none of those
    return music


def single_convert(filename):
    file = [[filename, 0]]
    p = create_piano_rolls(file)
    create_input(p, size=5000, save=True)


def single_save_roll(p_roll):
    new_mid = piano_roll_to_midi(p_roll)
    new_mid.save("output/output.mid")


def print_roll(p_roll):
    for i in range(len(p_roll)):
        for y in range(len(p_roll[i])):
            print("%.1f " % p_roll[i][y], end="")
        print("")


def clip_velocity(p_roll, act=0.2):
    n_roll = p_roll
    for i in range(len(p_roll)):
        for y in range(len(p_roll[i])):
            if p_roll[i][y] >= act:
                n_roll[i][y] = 1
            else:
                n_roll[i][y] = 0

    return n_roll


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
            velocity=90,
            time=int((time - now) * 20))
        track.append(message)
        now = time
    return midi


def create_piano_rolls(m):  # working !!
    music = m  # cycle_folders()
    # print(music[89])

    rolls = []

    for m in range(len(music)):
        mid = music[m][0]
        midi = pm.PrettyMIDI(mid)
        roll = pm.PrettyMIDI.get_piano_roll(midi, fs=32)  # 32 showed best results up to now
        roll = np.clip(roll, a_min=0, a_max=1)
        # im.imshow(s)
        # im.show()

        # print(mid)
        # new_mid = piano_roll_to_midi(roll)
        # new_mid.save("output/output.mid")
        rolls.append([roll, music[m][1]])
        # print(rolls)

    return rolls


def create_input(rolls, size=2000, save=False):
    features = []
    labels = []

    for roll in rolls:
        music = np.swapaxes(roll[0], 1, 0)
        slice_count = roll[0].shape[1] // size

        for i in range(0, slice_count * size, size):
            features.append(music[i:i + size])
            # label = [0 for i in range(1)]   Todo: 4 composers + None, now only valid or not valid [1], [0] as label
            # label[roll[1]] = 1
            label = [1]
            labels.append(label)

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    print(features.shape)

    if save:
        new_mid = piano_roll_to_midi(np.swapaxes(features[0], 1, 0))
        new_mid.save("output/output.mid")

    with open("data/training_data.file", "wb") as f:
        pickle.dump((features, labels), f)

    return (features, labels)


def reduce_dimensions(features):
    # Todo: does not work with sparse data! try truncatedsvd
    return
    pca = TruncatedSVD(10000)
    print(features.shape)
    nfeatures = np.reshape(features, (features.shape[0], features.shape[1] * features.shape[2]))
    print(nfeatures.shape)
    # nfeatures = np.swapaxes(nfeatures, 0, 1)
    # features = np.reshape(nfeatures, (features.shape[0], 2000, 128))
    new_features = pca.fit_transform(nfeatures)
    print(new_features.shape)
    old_features = pca.inverse_transform(new_features)
    print(old_features.shape)
    # old_features = np.swapaxes(old_features, 0, 1)
    old_features = np.reshape(old_features, (features.shape[0], 2000, 128))
    print(old_features.shape)

    s = piano_roll_to_midi(np.swapaxes(old_features[0], 1, 0))
    s.save("output/output.mid")

    return features


def load_data():
    print("loading data...")
    try:
        with open("data/training_data.file", "rb") as f:
            print("loading from data/training_data.file")
            data = pickle.load(f)

        features, labels = data[0], data[1]
        return features, labels

    except FileNotFoundError:
        print("creating piano rolls from midi files")
        music = cycle_folders()
        p = create_piano_rolls(music)
        data = create_input(p)

        features, labels = data[0], data[1]
        return features, labels


if __name__ == "__main__":
    # m = cycle_folders()
    # create_midi_rolls(m)
    # p = create_piano_rolls()
    # s = create_input(p)
    # features, labels = load_data()
    # print(features.shape)
    # single_convert_test("midi/albeniz/alb_se5_format0.mid")
    f, l = load_data()
    reduce_dimensions(f[0:8])
