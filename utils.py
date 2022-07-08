import os
import librosa
from tqdm import tqdm
import numpy as np
import pandas as pd


def load_dataset(directory: str) -> pd.DataFrame:
    '''
    load_dataset(directory='data')

    Parses all the audio files of a directory into a DataFrame.

    Parameters
    ----------
    directory : str
        The directory containing the audio files.

    Returns
    -------
    data : pd.DataFrame
        The parsed DataFrame containing data
        for all the audio files in the directory.
    '''
    files = os.listdir(directory)

    data = pd.DataFrame(
        columns=['filename', 'sample_index', 'features', 'label']
    )

    print("Loading dataset...")
    for f in tqdm(files):
        data = pd.concat(
            (data, get_data_from_file(directory, f)),
            ignore_index=True
        )

    return data


def get_data_from_file(directory: str, f: str) -> pd.DataFrame:
    '''
    get_data_from_file(directory='data', f='1_george_0.wav')

    Reads a file into a DataFrame containing the filename, sample_index,
    MFCC features and label of each sample of the audio.

    Parameters
    ----------
    directory : str
        The directory containing the audio files.
    f : str
        The name of the file

    Returns
    -------
    data : pd.DataFrame
        The parsed DataFrame containing data for the audio in the file.
    '''
    path = os.path.join(directory, f)
    signal, sr = librosa.load(path)

    data = pd.DataFrame(
        columns=['filename', 'sample_index', 'features', 'label']
    )

    samples = get_equal_samples(signal, sr)

    for sample_index in range(samples.shape[0]):
        sample = samples[sample_index]

        # MFCC features
        features = librosa.feature.mfcc(y=sample, n_mfcc=13)

        # 1st derivative features
        delta_features = librosa.feature.delta(features)

        # 2nd detivative features
        delta2_features = librosa.feature.delta(features, order=2)
        features = np.concatenate((features, delta_features, delta2_features))

        # Final feature vector
        features = features.reshape(-1)

        label = f.split('_')[0] if sample_index == 0 else None

        data.loc[len(data.index)] = [f, sample_index, features, label]

    return data


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
    duration : int, optional
        The duration of each sample in seconds. Default is 1.

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
