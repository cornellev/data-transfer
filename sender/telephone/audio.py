import numpy as np 
from scipy.io.wavfile import write 

# Each 4-bit string maps to a point in the 16-QAM constellation plot
def bits_to_symbols(bitstring):
    qam_map = {
        '0000': -3 - 3j, '0001': -3 - 1j, '0010': -3 + 3j, '0011': -3 + 1j,
        '0100': -1 - 3j, '0101': -1 - 1j, '0110': -1 + 3j, '0111': -1 + 1j,
        '1000':  3 - 3j, '1001':  3 - 1j, '1010':  3 + 3j, '1011':  3 + 1j,
        '1100':  1 - 3j, '1101':  1 - 1j, '1110':  1 + 3j, '1111':  1 + 1j
    }

    # If the length of the bitstring is not a multiple of 4, fill the end with 0s
    if len(bitstring) % 4 != 0:
        bitstring += '0' * (4 - len(bitstring) % 4)
        
    points = [qam_map[bitstring[i:i+4]] for i in range(0, len(bitstring), 4)]
    return points

# Turns a list of 16-QAM points to audio waves 
def qam_modulate(points, fc=4000, fs=44100, symbol_rate=500):
    samples_per_point = int (fs / symbol_rate)
    time = np.linspace(0, 1 / symbol_rate, samples_per_point, endpoint=False)
    audio_wave = []

    for p in points:
        # In-phase component (cosine) 
        i = np.real(p)
        # Quadrature component (sine) 
        q = np.imag(p)
        segment = i * np.cos(2 * np.pi * fc * time) - q * np.sin(2 * np.pi * fc * time)
        audio_wave.extend(segment)

    return np.array(audio_wave, dtype=np.float32)

# Creates .wav file containing the data in the original bitstring 
def create_wav (audio_wave, filename='data.wav', fs=44100):
    write(filename, fs, audio_wave)