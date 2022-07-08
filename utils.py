import os
import librosa
import numpy as np


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


def get_equal_samples(y: np.ndarray, sr: int, duration: int = 1) -> np.ndarray:
    '''
    get_equal_samples(y=y, sr=22050, duration=1)

    Splits a signal into smaller samples of equal duration.
    The last sample gets zero padded.

    Parameters
    ----------
    y : np.ndarray
        The initial signal.
    sr : int
        The sample rate.
    duration : int
        The duration of each sample in seconds.

    Returns
    -------
    samples : ndarray
        The final splitted samples with the specified duration.
    '''
    samples = []

    sample_size = duration * sr
    num_samples = len(y) // sample_size + 1

    sample_start = 0
    for _ in range(num_samples):
        sample = y[sample_start:sample_start + sample_size]
        sample_start += sample_size

        if len(sample) < sample_size:
            sample = zero_pad(sample, sample_size)

        samples.append(sample)

    return np.array(samples)


def zero_pad(y: np.ndarray, target_size: int) -> np.ndarray:
    '''
    zero_pad(y=np.array([1, 2, 3, 4]), target_size=10)

    Pads a signal with trailing zeros to reach the target size.

    Parameters
    ----------
    y : np.ndarray
        The initial signal.
    target_size : int
        The size of the zero padded signal

    Returns
    -------
    y : ndarray
        The zero padded signal.
    '''
    return np.concatenate((y, np.zeros(target_size - len(y))))
