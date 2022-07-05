import librosa


class SignalPreprocessor:
    '''Handles signal preprocessing.'''

    def apply_filters(self, signal):
        '''Applies filters to the given `signal`.'''
        # TODO Filters
        return signal

    def change_sample_rate(self, signal, sr, target_sr):
        '''Changes the `signal`'s `sample_rate`.'''
        return librosa.resample(signal, sr, target_sr), target_sr
