import sounddevice as sd 
import numpy as np
from telephone.audio import bits_to_symbols, qam_modulate, create_wav

# Saves and plays an audio file containing the data 
def process_and_play(avro_bytes):
    bitstring = ''.join(f'{byte:08b}' for byte in avro_bytes)
    symbols = bits_to_symbols(bitstring)
    waveform = qam_modulate(symbols, fc=4000, fs=44100, symbol_rate=500)

    create_wav(waveform, filename="data.wav", fs=44100)

    sd.play(np.array(waveform, dtype=np.float32), samplerate=44100)