from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import numpy as np
import librosa
import pickle

from preprocess import SignalPreprocessor
from digits import KNNDigitClassifier
from segmentation import BackgroundForegroundClassifier

from utils import load_dataset, get_features_from_signal


def parse_args():
    '''Parses command line args.'''
    parser = ArgumentParser()
    parser.add_argument("input", help="path to the input audio file")
    parser.add_argument(
        "-m", "--model", required=False, help="path to the trained model"
    )
    args = vars(parser.parse_args())
    return args['input'], args['model']


if __name__ == "__main__":
    # Parse args
    input_file, model_file = parse_args()

    # Input signal
    input_signal, sample_rate = librosa.load(input_file)

    # Signal preprocessing
    preprocessor = SignalPreprocessor()

    input_signal = preprocessor.apply_filters(input_signal, sample_rate)
    input_signal, sample_rate = preprocessor.change_sample_rate(
        input_signal, sample_rate, 6000
    )

    # MFCC, Delta and Delta-Delta Features
    features = get_features_from_signal(input_signal)

    # Background vs Foreground
    background_vs_foreground = BackgroundForegroundClassifier()
    digit_audio_list = background_vs_foreground.segment(input_signal)

    # Digit Classifier
    if model_file:
        digit_classifier = pickle.load(open(model_file, 'rb'))
    else:
        # Load the training dataset
        print('Loading training dataset...')
        data = load_dataset('data')
        data.dropna(inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            np.array(data.features.values.tolist()),
            np.array(data.label),
            test_size=0.2
        )
        print('Training dataset loaded.')

        # Traing the digit classifier
        digit_classifier = KNNDigitClassifier(n_neighbors=5)
        digit_classifier.fit(X_train, y_train)

    digits = [digit_classifier.predict(audio) for audio in digit_audio_list]
    print(digits)
