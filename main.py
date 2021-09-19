from argparse import ArgumentParser
import librosa
import pickle

from preprocess import SignalPreprocessor
from digits import DigitClassifier
from segmentation import BackgroundForegroundClassifier

def parse_args():
    '''Parses command line args.'''
    parser = ArgumentParser()
    parser.add_argument("input", help="path to the input audio file")
    parser.add_argument("-m", "--model", required=False, help="path to the trained model")
    args = vars(parser.parse_args())
    return args['input'], args['model']

if __name__ == "__main__":
    # Parse args
    input_file, model_file = parse_args()

    # Input signal
    input_signal, sample_rate = librosa.load(input_file)

    # Signal preprocessing
    preprocessor = SignalPreprocessor()

    input_signal = preprocessor.apply_filters(input_signal)
    input_signal, sample_rate = preprocessor.change_sample_rate(input_signal, sample_rate) # TODO Probably change

    # Background vs Foreground
    background_vs_foreground = BackgroundForegroundClassifier()
    digit_audio_list = background_vs_foreground.segment(input_signal)
    
    # Digit Classifier
    if model_file:
        digit_classifier = pickle.load(open(model_file, 'rb'))
    else:
        digit_classifier = DigitClassifier()
        digit_classifier.fit([]) # TODO Train classifier with spoken digits dataset

    digits = [digit_classifier.predict(audio) for audio in digit_audio_list]
    print(digits)

    # TODO Calculate accuracy
