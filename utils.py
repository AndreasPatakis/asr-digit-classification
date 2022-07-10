import os
import librosa
from tqdm import tqdm
import numpy as np
import pandas as pd
from random import choice

from preprocess import SignalPreprocessor

sp = SignalPreprocessor()


def load_dataset(directory: str, background: bool = False) -> pd.DataFrame:
    '''
    load_dataset(directory='data')

    Parses all the audio files of a directory into a DataFrame.

    Parameters
    ----------
    directory : str
        The directory containing the audio files.
    background : bool, optional
        Indicates whether the directory contains background audio files.
        If True all samples are labeled as "background".

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

    print(f'Loading dataset from "{directory}"...')
    for f in tqdm(files):
        data = pd.concat(
            (data, get_data_from_file(directory, f, background)),
            ignore_index=True
        )

    return data


def get_data_from_file(
    directory: str,
    f: str,
    background: bool = False
) -> pd.DataFrame:
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
    background : bool, optional
        Indicates whether the directory contains background audio files.
        If True all samples are labeled as "background".

    Returns
    -------
    data : pd.DataFrame
        The parsed DataFrame containing data for the audio in the file.
    '''
    path = os.path.join(directory, f)
    signal, sr = librosa.load(path)

    # Preprocess the signal

    # Change sample rate to 8KHz
    signal, sr = sp.change_sample_rate(signal, sr, 8000)

    # Apply filters to keep only the fundamental frequencies
    # of the human voice.
    signal = sp.apply_filters(signal, sr)

    data = pd.DataFrame(
        columns=['filename', 'sample_index', 'features', 'label']
    )

    samples = get_equal_samples(signal, sr)

    for sample_index in range(samples.shape[0]):
        features = get_features_from_signal(samples[sample_index])

        if background:
            label = 'background'
        else:
            label = f.split('_')[0] if sample_index == 0 else None

        data.loc[len(data.index)] = [f, sample_index, features, label]

    return data


def get_features_from_signal(y: np.ndarray):
    '''
    get_features_from_signal(y=y)

    Extracts MFCC, Delta and Delta-Delta features from
    the given signal. This results in a total of 39 features.

    Parameters
    ----------
    y : np.ndarray
        The signal to extract features from.

    Returns
    -------
    features : ndarray
        The extracted features.
    '''
    # MFCC features
    features = librosa.feature.mfcc(y=y, n_mfcc=13)

    # 1st derivative features
    delta_features = librosa.feature.delta(features)

    # 2nd detivative features
    delta2_features = librosa.feature.delta(features, order=2)
    features = np.concatenate((features, delta_features, delta2_features))

    # Return the final feature vector
    return features.reshape(-1)


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


def create_noisy_dataset(
    background_directory: str,
    foreground_directory: str,
    data_size: int
) -> pd.DataFrame:
    '''
    create_noisy_dataset('data/background', 'data/digits', 1000)

    Mixes background and foreground signals
    to create a dataset of noisy signals.

    Parameters
    ----------
    background_directory : str
        The directory containing the background audio files.
    foreground_directory : str
        The directory containing the foreground audio files.
    data_size : int
        The number of noisy signals in the noisy dataset.

    Returns
    -------
    data : pd.DataFrame
        The parsed DataFrame containing data for the created noisy signals.
    '''
    data = pd.DataFrame(
        columns=['filename', 'sample_index', 'features', 'label']
    )

    background_files = os.listdir(background_directory)
    foreground_files = os.listdir(foreground_directory)

    print("Creating noisy dataset...")
    for _ in tqdm(range(data_size)):
        background_file = choice(background_files)
        background_path = os.path.join(background_directory, background_file)

        foreground_file = choice(foreground_files)
        foreground_path = os.path.join(foreground_directory, foreground_file)

        background, bsr = librosa.load(background_path)
        foreground, fsr = librosa.load(foreground_path)

        background_signal = choice(get_equal_samples(background, bsr))
        foreground_signal = get_equal_samples(foreground, fsr)[0]

        noisy_signal = 15 * background_signal + foreground_signal

        # Preprocess the noisy signal

        # Change sample rate to 8KHz
        noisy_signal, nsr = sp.change_sample_rate(noisy_signal, bsr, 8000)

        # Apply filters to keep only the fundamental frequencies
        # of the human voice.
        noisy_signal = sp.apply_filters(noisy_signal, nsr)

        data.loc[len(data.index)] = [
            f'{background_file} + {foreground_file}',
            0,
            get_features_from_signal(noisy_signal),
            'foreground'
        ]

    return data
