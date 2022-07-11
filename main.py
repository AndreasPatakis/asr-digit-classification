from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import librosa
import pickle

from preprocess import apply_filters, change_sample_rate
from segmentation import SVMBackgroundForegroundClassifier, NNClassifier
from digits import KNNDigitClassifier

from utils import get_equal_samples, load_dataset, get_features_from_signal,\
    create_noisy_dataset


def parse_args():
    '''Parses command line args.'''
    parser = ArgumentParser()
    parser.add_argument("input", help="path to the input audio file")
    args = vars(parser.parse_args())
    return args['input']


def train_background_vs_foreground() -> SVMBackgroundForegroundClassifier:
    '''
    Trains a new instance of SVMBackgroundForegroundClassifier.

    Returns
    -------
    classifier : SVMBackgroundForegroundClassifier
        The trained instance of the background vs foreground classifier.
    '''
    try:
        with open("foreground_dataset.pickle", "rb") as f:
            foreground_data = pickle.load(f)
    except FileNotFoundError:
        foreground_data = create_noisy_dataset(
            'data/background',
            'data/digits',
            2000
        )

    try:
        with open("background_dataset.pickle", "rb") as f:
            background_dataset = pickle.load(f)
    except FileNotFoundError:
        background_data = load_dataset('data/background', background=True)

    data = pd.concat((background_data, foreground_data), ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(data.features.values.tolist()),
        np.array(data.label),
        test_size=0.2
    )

    classifier = SVMBackgroundForegroundClassifier()

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Calculate the classifier's score
    score = classifier.score(X_test, y_test)
    print(f'Score: {score}')

    # Save the classifier
    classifier.save()

    return classifier


def train_digit_classifier() -> NNClassifier:
    '''
    Trains a new instance of NNClassifier.

    Returns
    -------
    classifier : NNClassifier
        The trained instance of the digit classifier.
    '''
    try:
        with open("noisy_foreground_dataset.pickle", "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        data = load_dataset('data/digits')
        data.dropna(inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(data.features.values.tolist()),
        np.array(data.label),
        test_size=0.2
    )

    classifier = NNClassifier()

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Calculate the classifier's score
    score = classifier.score(X_test, y_test)
    print(f'Score: {score}')

    # Save the classifier
    classifier.save()

    return classifier


if __name__ == "__main__":
    # Parse args
    input_file = parse_args()

    # Input signal
    input_signal, sample_rate = librosa.load(input_file)

    # Signal preprocessing

    # Change sample rate to 8KHz
    input_signal, sample_rate = change_sample_rate(
        input_signal, sample_rate, 8000
    )

    # Apply filters to keep only the fundamental frequencies
    # of the human voice.
    input_signal = apply_filters(input_signal, sample_rate)

    input_samples = get_equal_samples(input_signal, sample_rate)

    # Collect features
    input_features = []

    for sample in input_samples:
        # MFCC, Delta and Delta-Delta Features
        features = get_features_from_signal(sample)

        input_features.append(features)


    # Background vs Foreground
    try:
        with open("svm_bf_model.pickle", "rb") as f:
            background_vs_foreground = pickle.load(f)
    except FileNotFoundError:
        background_vs_foreground = train_background_vs_foreground()

    bf_labels = background_vs_foreground.predict(input_features)
    bf_data = pd.DataFrame({'features': input_features, 'label': bf_labels})

    data = bf_data[bf_data.label == 'foreground']

    # Digit Classifier
    try:
        with open("nn_digit_model.pickle", "rb") as f:
            digit_classifier = pickle.load(f)
    except FileNotFoundError:
        digit_classifier = train_digit_classifier()

    digits = digit_classifier.predict(
        np.array(data.features.values.tolist()),
    )
    print('Foreground vs Background classifier recognised {} instances of human, foreground sound.'.format(len(digits)))
    print('Digit prediction: {}'.format(' '.join(digits)))
