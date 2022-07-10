import os
import random
import librosa
import utils
import numpy as np
from scipy.io.wavfile import write

from preprocess import SignalPreprocessor

sp = SignalPreprocessor()


def read_folder(path:str):
    return sorted(os.listdir(path))

def split_digits(data, chunk=300):
    digits = []
    for i in range(0,len(data),chunk):
        digits.append(data[i:i+chunk])
    return digits

def get_digit(digits_list,digit):
    samples = len(digits_list[0])
    return random.randint(0,samples)

def get_n_length_sample(signal,sr,duration=1,random_start=False):
    sample_size = sr*duration
    sample_start = 0 if not random_start else random.randint(0,len(signal)-sample_size)
    sample = signal[sample_start:sample_start+sample_size]
    if len(sample)<sample_size:
        sample = utils.zero_pad(sample,sample_size)
    return sample



def get_noise_signal(directory,duration):

    noise_files = read_folder(directory)
    #noise_file_name = noise_files[random.randint(0,len(noise_files))]

    signal, sr = librosa.load(os.path.join(directory,noise_files[100]))
    #Amplify noise signal a bit
    signal *= 2

    # Change sample rate to 8KHz
    signal, sr = sp.change_sample_rate(signal, sr, 8000)


    return get_n_length_sample(signal,sr,int(duration),random_start=True), sr

def get_digit_signal(digit_file,noise_signal):

    signal, sr = librosa.load(digit_file)

    # Change sample rate to 8KHz
    signal, sr = sp.change_sample_rate(signal, sr, 8000)

    signal = get_n_length_sample(signal,sr)

    #Adding noise to the recording
    signal = np.add(signal,noise_signal[:sr]*0.7)

    return signal, sr


def get_audio(digits_dir,noise_dir):
    digit_values = [x for x in range(10)]
    digits = split_digits(read_folder(digits_dir))
    final_signal = np.empty(1, dtype=np.int8)
    while digit_values:
        while True:
            curr_digit = random.randint(digit_values[0],digit_values[-1])
            if curr_digit in digit_values:
                digit_values.pop(digit_values.index(curr_digit))
                break
        print("Number: ",curr_digit)
        #Both noise and digit signals are resampled to 8KHz sample rate, so sr is the same
        #Noise duration will be between 1 to 4 seconds long
        noise_duration = random.randint(1,2)
        noise_signal, n_sr = get_noise_signal(noise_dir,noise_duration)

        #Pick a random recording of the chosen digit
        recording = random.randint(0,len(digits[0]))
        digit_recording = digits[curr_digit][recording]
        digit_signal, d_sr = get_digit_signal(os.path.join(digits_dir,digit_recording),noise_signal)

        final_signal = np.append(final_signal,noise_signal)
        final_signal = np.append(final_signal,digit_signal)


    final_signal = np.delete(final_signal,0)
    write("example.wav", d_sr, final_signal)

    return final_signal, d_sr


if __name__ == '__main__':
    digits_dir = './data/digits'
    noise_dir = './data/background'

    signal, sr = get_audio(digits_dir,noise_dir)
