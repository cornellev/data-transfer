"""
Testing the end-to-end functionality of the sender and receiver, where we encode something and decode on the receiver.
Used to compare a lot of different configurations and implementations.
"""

from sender.telephone.audio import *

from receiver.telephone.receiver import *
from receiver.telephone.audio_decoder import *

fc = 4000
fs = 44100
symbol_rate = 100

# Testing the bits being sent using the wav bitstreaming
bits = "0101010101010101010101010101010101010101010101010101010101010101"

def calculate_error_rate(original_bits, received_bits):
    """
    Calculate the error rate between the original bits and the received bits.
    """
    min_length = min(len(original_bits), len(received_bits))
    errors = sum(1 for a, b in zip(original_bits[:min_length], received_bits[:min_length]) if a != b)
    ber = errors / min_length

    # Print results
    print(f"Original bits length: {len(original_bits)}")
    print(f"Recovered bits length: {len(received_bits)}")
    print(f"Compared first {min_length} bits")
    print(f"Bit errors: {errors}")
    print(f"Bit Error Rate (BER): {ber:.6f}")
    
    return ber

symbols = bits_to_symbols(bits)
# print("Symbols: ", symbols)
audio_wave = qam_modulate(symbols, fc, fs, symbol_rate)

audio_wave = audio_wave / np.max(np.abs(audio_wave))
audio_wave = (audio_wave * 32767).astype(np.int16)

with open("wave_raw.txt", "w") as f:
    for i in range(len(audio_wave)):
        f.write(str(audio_wave[i]) + "\n")

# now stream the audio_wave to the demodulator
demodulator = BufferedDemodulator(fc, fs, symbol_rate)
demodulator.reset()

recovered_bits = []
# send the audio_wave to the demodulator
for i in range(0, len(audio_wave), 1024):
    chunk = audio_wave[i:i+1024]
    demodulated_chunk = demodulator.demodulate(chunk)
    recovered_bits.extend(demodulated_chunk)
# wait for the demodulator to finish

recovered_bits = "".join([str(bit) for bit in recovered_bits])

print(bits)
print(recovered_bits)
calculate_error_rate(bits, recovered_bits)

# calculate_error_rate(bits, QAMDemodulator(fc, fs, symbol_rate).demodulate(audio_wave))


# Strings - encode some string into bits and send and decode

# Making the telephone call