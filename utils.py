import os
import librosa


def load_dataset(directory):
    files = os.listdir(directory)

    features = list(
        map(lambda f: get_features_from_file(directory, f), files)
    )

    labels = list(
        map(lambda f: f.split('_')[0], files)
    )

    return features, labels


def get_features_from_file(directory, f):
    path = os.path.join(directory, f)
    signal = librosa.load(path)[0]
    return librosa.feature.mfcc(y=signal)
