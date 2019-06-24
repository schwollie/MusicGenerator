import numpy as np

import midiHandler
import model

if __name__ == "__main__":
    features, labels = midiHandler.load_data()
    # labels not needed right now because they are all valid [1]

    shape = features.shape
    features = np.reshape(features, (shape[0], shape[2], shape[1], 1))
    print(features.shape)

    net = model.AutoEncoder()
    net.train(features[0:2], epochs=20)

    e = net.encode(features[0:1])
    # midiHandler.print_roll(features[0])
    d = net.decode(e)

    mse = np.square(np.subtract(features[0:1], d)).mean()
    print(mse)
    print(d.shape)
    d = np.squeeze(d)

    d = midiHandler.clip_velocity(d, act=0.9)
    # midiHandler.print_roll(d)
    midiHandler.single_save_roll(d)
    print(d.shape)
    # midiHandler.print_roll(d)

    # net.train(features)
    # network = model.GAN()
    # network.train(features, epochs=1)
