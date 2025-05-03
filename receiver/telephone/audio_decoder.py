"""
Defines the AudioDecoder system using 16-QAM modulation. We need to decode a stream of audio data into bits.

Testing plan:
1. Rely on the encoder to generate the audio for a certain bitstring.
2. Decode the audio back into bits and compare with the original bitstring to calculate the bit error rate.
"""

import numpy as np
from scipy.io.wavfile import read
from scipy import signal


class QAMDemodulator:
    """
    Symbol classifier for 16-QAM signals.
    Maps complex I/Q samples to 4-bit strings.
    """
    def __init__(self):
        # 16-QAM constellation
        self.qam_map = {
            (-3, -3): '0000', (-3, -1): '0001', (-3, 3): '0010', (-3, 1): '0011',
            (-1, -3): '0100', (-1, -1): '0101', (-1, 3): '0110', (-1, 1): '0111',
            (3, -3): '1000', (3, -1): '1001', (3, 3): '1010', (3, 1): '1011',
            (1, -3): '1100', (1, -1): '1101', (1, 3): '1110', (1, 1): '1111'
        }

        self.constellation_points = np.array(
            [complex(i, q) for (i, q) in self.qam_map.keys()]
        )
        self.inverse_map = {
            point: bits for point, bits in zip(self.constellation_points, self.qam_map.values())
        }

    def classify(self, symbol: complex) -> str:
        """
        Classify a complex symbol to the closest 16-QAM point.

        Args:
            symbol: complex number (I + jQ)

        Returns:
            4-bit string
        """
        closest_idx = np.argmin(np.abs(self.constellation_points - symbol))
        closest_point = self.constellation_points[closest_idx]
        return self.inverse_map[closest_point]

class BufferedDemodulator:
    """
    Streaming 16-QAM demodulator that processes audio chunks in real-time.
    Performs carrier mixing, low-pass filtering, and symbol sampling.
    """
    def __init__(self, carrier_freq=4000, sample_rate=44100, symbol_rate=100):
        self.fc = carrier_freq
        self.fs = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = int(self.fs / self.symbol_rate)

        # Create 16-QAM constellation mapping
        self.qam_map = {
            (-3, -3): '0000', (-3, -1): '0001', (-3, 3): '0010', (-3, 1): '0011',
            (-1, -3): '0100', (-1, -1): '0101', (-1, 3): '0110', (-1, 1): '0111',
            (3, -3): '1000', (3, -1): '1001', (3, 3): '1010', (3, 1): '1011',
            (1, -3): '1100', (1, -1): '1101', (1, 3): '1110', (1, 1): '1111'
        }
        self.constellation_points = np.array([complex(i, q) for (i, q) in self.qam_map.keys()])

        # Signal buffer
        self.buffer = np.array([], dtype=np.float32)
        self.total_samples_seen = 0

        # Design low-pass filter once (cutoff slightly above symbol rate)
        nyquist = self.fs / 2
        cutoff = self.symbol_rate * 1.5 / nyquist
        self.lpf_b, self.lpf_a = signal.butter(5, cutoff)

        # Filter state across chunks
        self.i_filter_state = None
        self.q_filter_state = None

    def reset(self):
        """Reset internal state for fresh decoding."""
        self.buffer = np.array([], dtype=np.float32)
        self.total_samples_seen = 0
        self.i_filter_state = None
        self.q_filter_state = None

    def demodulate(self, chunk: np.ndarray) -> list:
        """
        Demodulate a chunk of audio samples into bit list.

        Args:
            chunk: np.ndarray of int16 or float32 audio samples

        Returns:
            List of recovered bits (as ints 0/1)
        """
        # Normalize and convert to float32 if necessary
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
            chunk /= np.max(np.abs(chunk))  # normalize to [-1, 1]

        # Append to rolling buffer
        self.buffer = np.append(self.buffer, chunk)

        bits_out = []

        # Process full symbol-sized windows
        while len(self.buffer) >= self.samples_per_symbol:
            # Take one full symbol window
            symbol_samples = self.buffer[:self.samples_per_symbol]

            # Time vector for this symbol, using global sample index
            t_offset = self.total_samples_seen
            t = (np.arange(self.samples_per_symbol) + self.total_samples_seen) / self.fs

            # Mix down to baseband
            i_carrier = np.cos(2 * np.pi * self.fc * t)
            q_carrier = -np.sin(2 * np.pi * self.fc * t)
            i_mixed = symbol_samples * i_carrier
            q_mixed = symbol_samples * q_carrier

            # Apply low-pass filter (preserving state)
            i_filtered, self.i_filter_state = signal.lfilter(
                self.lpf_b, self.lpf_a, i_mixed, zi=self._init_filter_state(self.i_filter_state, i_mixed))
            q_filtered, self.q_filter_state = signal.lfilter(
                self.lpf_b, self.lpf_a, q_mixed, zi=self._init_filter_state(self.q_filter_state, q_mixed))

            # Sample at center
            mid = self.samples_per_symbol // 2
            i_sample = i_filtered[mid]
            q_sample = q_filtered[mid]
            received_symbol = complex(i_sample, q_sample)

            # Normalize to constellation scale (max ±3)
            scale = max(np.abs(i_sample), np.abs(q_sample)) / 3.0
            if scale > 0:
                received_symbol /= scale

            # Classify symbol to nearest constellation point
            nearest_idx = np.argmin(np.abs(self.constellation_points - received_symbol))
            symbol_point = self.constellation_points[nearest_idx]
            bits = self.qam_map[(int(np.real(symbol_point)), int(np.imag(symbol_point)))]
            bits_out.extend([int(b) for b in bits])

            # Update state
            self.buffer = self.buffer[self.samples_per_symbol:]
            self.total_samples_seen += self.samples_per_symbol

        return bits_out

    def _init_filter_state(self, previous_state, signal_chunk):
        """
        Initialize or propagate filter state for streaming `lfilter`.

        Args:
            previous_state: previous zi (or None)
            signal_chunk: current signal to be filtered

        Returns:
            zi for lfilter
        """
        if previous_state is not None:
            return previous_state
        zi = signal.lfilter_zi(self.lpf_b, self.lpf_a) * signal_chunk[0]
        return zi



# Example usage demonstrating the decoder
if __name__ == "__main__":
    
    # Encode using the provided encoder (assuming it's imported)
    from sender.telephone.audio import bits_to_symbols, qam_modulate, create_wav
    fc = 4000
    fs = 44100
    symbol_rate = 100
    
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
    
    def test_wav_file():
        # Test bitstring
        # test_bits = "1010110010110010101110001111000001101010100010101100010101001100101010100101010101010100010101010001010101010010101010101100101011100011110000011010101000101011000101010011001010101001010101010101000101010100010110010101110001111000001101010100010101100010101001100101010100101010101010100010101010001010101010010101010101011001010111000111100000110101010001010110001010100110010101010010101010101010001010101000101010101001010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101011100011110000011010101000101011000101010011001010101001010101010101000101010100010101010100101010101010110110010101110001111000001101010100010101100010101001100101010100101010101010100010101010001010101010010101010101010101010101010101010101010101010101010101010101010010101010101010101010101010101010101010101010101011001010111000111100000110101010001010110001010100110010101010010101010101010001010101000101010101001010101010110010101110001111000001101010100010101100010101001100101010100101010101010100010101010001010101010010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010"

        # message = "I'd be happy to help you convert that hexadecimal representation back into binary bits (0s and 1s)."
        message = "hello"
    
        # turn this message into bits in a string format
        bits = ''.join(format(ord(c), '08b') for c in message)
        test_bits = bits
        
        # Convert bits to symbols
        symbols = bits_to_symbols(test_bits)
        
        # Modulate symbols to audio
        audio_wave = qam_modulate(symbols, fc=fc, fs=fs, symbol_rate=symbol_rate)
        
        # Create WAV file
        create_wav(audio_wave, "test_qam.wav")
        
        print(f"Original bitstring: {test_bits}")
        
        # Now demodulate using our decoder
        qam_demod = QAMDemodulator(carrier_freq=fc, sample_rate=fs, symbol_rate=symbol_rate)
        decoded_bits = qam_demod.demodulate_file("test_qam.wav")
        
        print(f"Decoded bitstring: {decoded_bits}")
        
        # Calculate bit error rate
        errors = sum(1 for a, b in zip(test_bits, decoded_bits) if a != b)
        ber = errors / len(test_bits)
        print(f"Bit error rate: {ber:.6f} ({errors} errors out of {len(test_bits)} bits)")
    
    def test_buffered_demodulator():
        # test_bits = "1010110010110010101110001111000001101010100010101100010101001100" * 10  # make it a bit longer
        test_bits = "0101101010101000101101010101010101010101010101010101010101010101010101010101010101010101010101010101010101"
        # test_bits = "01010101100101010101001101001110101010101010010110101010101010101010101010101010101"

        # Encode: bits -> symbols -> modulate -> audio wave
        symbols = bits_to_symbols(test_bits)
        audio_wave = qam_modulate(symbols, fc=fc, fs=fs, symbol_rate=symbol_rate)

        # Simulate 16-bit audio format (as if from microphone)
        audio_wave = audio_wave / np.max(np.abs(audio_wave))
        audio_wave_int16 = (audio_wave * 32767).astype(np.int16)
        
        with open("wave_raw_right.txt", "w") as f:
            for i in range(len(audio_wave_int16)):
                f.write(str(audio_wave_int16[i]) + "\n")

        # Initialize enhanced demodulator
        enhanced_demod = BufferedDemodulator(carrier_freq=fc, sample_rate=fs, symbol_rate=symbol_rate)

        # Gradually feed chunks and collect decoded bits
        recovered_bits = []
        
        chunk_size = 1024  # Process in chunks of 1024 samples

        for i in range(0, len(audio_wave_int16), chunk_size):
            chunk = audio_wave_int16[i:i+chunk_size]
            bits = enhanced_demod.demodulate(chunk)
            recovered_bits.extend(bits)
        
        # pass in the entire buffer
        # print(f"Received {len(audio_wave_int16)} samples")
        
        # bits = enhanced_demod.demodulate(audio_wave_int16)
        # recovered_bits.extend(bits)

        # Join recovered bits into a string
        recovered_bitstring = ''.join(str(bit) for bit in recovered_bits)
        
        # print out the two bitstrings to compare
        print(f"Original bitstring: {test_bits}")
        print(f"Recovered bitstring: {recovered_bitstring}")

        # Compare
        calculate_error_rate(test_bits, recovered_bitstring)
        # Reset the demodulator for next use
        enhanced_demod.reset()
    
    # test_wav_file()
    test_buffered_demodulator()
