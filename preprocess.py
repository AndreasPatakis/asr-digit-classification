import librosa
import librosa.display
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt


class SignalPreprocessor:
    '''Handles signal preprocessing.'''

    def apply_filters(self, signal, sr):
        '''
        apply_filters(self, signal=y)

        Applies preprocessing filters to the given `signal`.

        Parameters
        ----------
        signal : np.ndarray
            The signal to be filtered.

        sr: sr
            The sample rate of the signal.

        Returns
        ----------
        filtered : np.ndarray
            The filtered signal.
        '''
        # Female voice
        # Fundamental frequency: 350Hz to 3KHz
        # Harmonics :3KHz to 17KHz

        # Male voice
        # Fundamental frequency: 100Hz to 900Hz
        # Harmonics :900Hz to 8KHz

        # Bandpass filter
        # 300Hz - 3KHz (Keep fundamental freuqncies, ignore harmonics)

        return self.apply_bandpass_filter(signal, sr, 300, 3000)

    def apply_bandpass_filter(self, signal, sr, low, high):
        '''
        apply_bandpass_filter(self, signal, 300, 3000)

        Applies a bandpass filter to the given `signal`.

        Parameters
        ----------
        signal : np.ndarray
            The signal to be filtered.

        sr: sr
            The sample rate of the signal.

        low: int
            The lowcut freuency of the filter.

        high: int
            The highcut freuency of the filter.

        Returns
        ----------
        filtered : np.ndarray
            The filtered signal.
        '''
        b, a = butter(2, [low, high], 'bandpass', analog=False, fs=sr)
        return lfilter(b, a, signal)

    def change_sample_rate(self, signal, sr, target_sr=6000):
        '''
        fit(signal=y, sr=22050, target_sr=6000)

        Changes the `signal`'s `sample_rate`.

        Parameters
        ----------
        signal : np.ndarray
            The signal to be resampled.

        sr: sr
            The initial sample rate of the signal.

        target_sr: target_sr, optional
            The target sample rate of the signal.
            Defaults to 6KHz which is twice the
            Nyqist frequency for human voice (3KHz).
        '''
        return librosa.resample(signal, orig_sr=sr, target_sr=target_sr),\
            target_sr
