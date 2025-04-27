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
    16-QAM demodulator that converts modulated signals back to bitstreams.
    """
    def __init__(self, carrier_freq=4000, sample_rate=44100, symbol_rate=500):
        """
        Initialize the QAM demodulator.
        
        Args:
            carrier_freq: Carrier frequency in Hz
            sample_rate: Sampling rate in Hz
            symbol_rate: Symbol rate in symbols per second
        """
        self.fc = carrier_freq
        self.fs = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = int(self.fs / self.symbol_rate)
        
        # 16-QAM constellation mapping
        self.qam_map = {
            (-3, -3): '0000', (-3, -1): '0001', (-3, 3): '0010', (-3, 1): '0011',
            (-1, -3): '0100', (-1, -1): '0101', (-1, 3): '0110', (-1, 1): '0111',
            (3, -3): '1000', (3, -1): '1001', (3, 3): '1010', (3, 1): '1011',
            (1, -3): '1100', (1, -1): '1101', (1, 3): '1110', (1, 1): '1111'
        }
        
        # Create reference constellation points
        self.constellation_points = np.array([complex(i, q) for (i, q) in self.qam_map.keys()])
        
    def demodulate_file(self, filename):
        """
        Demodulate QAM from a WAV file.
        
        Args:
            filename: Path to WAV file
            
        Returns:
            Demodulated bitstring
        """
        # Read the WAV file
        fs, audio_data = read(filename)
        
        # Ensure data is float for processing
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # Normalize if necessary (if data is in int16 format)
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / 32768.0
            
        return self.demodulate(audio_data)
    
    def demodulate(self, audio_signal):
        """
        Demodulate 16-QAM signal to bitstring.
        
        Args:
            audio_signal: Audio signal as numpy array
            
        Returns:
            Demodulated bitstring
        """
        # Create time base for the entire signal
        t = np.arange(len(audio_signal)) / self.fs
        
        # Generate in-phase and quadrature carriers
        i_carrier = np.cos(2 * np.pi * self.fc * t)
        q_carrier = -np.sin(2 * np.pi * self.fc * t)  # Negative sine for quadrature
        
        # Demodulate I and Q components
        i_demod = audio_signal * i_carrier
        q_demod = audio_signal * q_carrier
        
        # Low-pass filter to remove the 2*fc component
        # Design a lowpass filter with cutoff at symbol_rate
        nyquist = self.fs / 2
        cutoff = self.symbol_rate * 1.5  # Slightly higher than symbol rate for good reception
        b, a = signal.butter(5, cutoff / nyquist)
        
        # Apply the filter
        i_filtered = signal.filtfilt(b, a, i_demod)
        q_filtered = signal.filtfilt(b, a, q_demod)
        
        # Sample at the center of each symbol
        num_symbols = len(audio_signal) // self.samples_per_symbol
        i_sampled = np.zeros(num_symbols)
        q_sampled = np.zeros(num_symbols)
        
        for i in range(num_symbols):
            # Sample near the middle of each symbol period
            sample_point = i * self.samples_per_symbol + self.samples_per_symbol // 2
            if sample_point < len(i_filtered):
                i_sampled[i] = i_filtered[sample_point]
                q_sampled[i] = q_filtered[sample_point]
        
        # Normalization factor (based on the maximum expected constellation value)
        # Original encoder uses values up to ±3, so we scale accordingly
        scale_factor = np.max(np.abs(np.concatenate([i_sampled, q_sampled]))) / 3.0
        if scale_factor > 0:
            i_sampled = i_sampled / scale_factor
            q_sampled = q_sampled / scale_factor
        
        # Combine into complex numbers
        received_symbols = i_sampled + 1j * q_sampled
        
        # Symbol decision - map each received point to the nearest constellation point
        decided_bits = []
        
        for symbol in received_symbols:
            # Find the closest constellation point
            distances = np.abs(self.constellation_points - symbol)
            closest_idx = np.argmin(distances)
            closest_point = self.constellation_points[closest_idx]
            
            # Map back to bits
            i_val = int(np.real(closest_point))
            q_val = int(np.imag(closest_point))
            bit_value = self.qam_map.get((i_val, q_val), '0000')  # Default to '0000' if not found
            decided_bits.append(bit_value)
        
        # Join all bits together
        bitstring = ''.join(decided_bits)
        
        return bitstring

class BufferedDemodulator:
    """
    Wrapper class to handle audio chunks being sent in as a data stream.
    """
    def __init__(self, carrier_freq=4000, sample_rate=44100, symbol_rate=500):
        self.qam_demodulator = QAMDemodulator(carrier_freq, sample_rate, symbol_rate)
        self.buffer = np.array([], dtype=np.float32)
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = int(sample_rate / symbol_rate)
        self.total_samples_seen = 0  # To track global time across chunks

    def demodulate(self, digital_signal: np.ndarray) -> list:
        """
        Demodulate the digital signal into bits.
        
        Args:
            digital_signal: Digital signal as numpy array (int16)
            
        Returns:
            List of demodulated bits (0 or 1)
        """
        # Convert int16 input to float32
        float_signal = digital_signal.astype(np.float32) / 32768.0
        
        print(f'Received {len(float_signal)} samples')
        
        # Append new chunk to buffer
        self.buffer = np.append(self.buffer, float_signal)
        
        bits = []
        while len(self.buffer) >= self.samples_per_symbol:
            # Only take exactly one symbol worth of samples at a time
            symbol_samples = self.buffer[:self.samples_per_symbol]
            
            # Process one symbol at a time
            bitstring = self.qam_demodulator.demodulate(symbol_samples)
            
            for bit in bitstring:
                bits.append(int(bit))
            
            # Remove processed samples
            self.buffer = self.buffer[self.samples_per_symbol:]
            
            # Advance sample counter (simulate continuous time)
            self.total_samples_seen += self.samples_per_symbol
        
        print(f"returning {len(bits)} bits")
        return bits

    def reset(self):
        """Reset the demodulator state."""
        self.buffer = np.array([], dtype=np.float32)
        self.total_samples_seen = 0


# Example usage demonstrating the decoder
if __name__ == "__main__":
    # Test bitstring
    test_bits = "10110010101110001111000001101010100010101100010101001100101010100101010101010100010101010001010101010010101010101010101010101010101010101010101010101010101010101010"
    
    # Encode using the provided encoder (assuming it's imported)
    from sender.telephone.audio import bits_to_symbols, qam_modulate, create_wav
    fc = 4000
    fs = 44100
    symbol_rate = 100
    
    def test_wav_file():
        # Test bitstring
        test_bits = "1010110010110010101110001111000001101010100010101100010101001100101010100101010101010100010101010001010101010010101010101100101011100011110000011010101000101011000101010011001010101001010101010101000101010100010110010101110001111000001101010100010101100010101001100101010100101010101010100010101010001010101010010101010101011001010111000111100000110101010001010110001010100110010101010010101010101010001010101000101010101001010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101011100011110000011010101000101011000101010011001010101001010101010101000101010100010101010100101010101010110110010101110001111000001101010100010101100010101001100101010100101010101010100010101010001010101010010101010101010101010101010101010101010101010101010101010101010010101010101010101010101010101010101010101010101011001010111000111100000110101010001010110001010100110010101010010101010101010001010101000101010101001010101010110010101110001111000001101010100010101100010101001100101010100101010101010100010101010001010101010010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010"
        
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
        test_bits = "1010110010110010101110001111000001101010100010101100010101001100" * 10  # make it a bit longer

        # Encode: bits -> symbols -> modulate -> audio wave
        symbols = bits_to_symbols(test_bits)
        audio_wave = qam_modulate(symbols, fc=fc, fs=fs, symbol_rate=symbol_rate)

        # Simulate 16-bit audio format (as if from microphone)
        audio_wave = audio_wave / np.max(np.abs(audio_wave))
        audio_wave_int16 = (audio_wave * 32767).astype(np.int16)

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

        # Compare
        min_length = min(len(test_bits), len(recovered_bitstring))
        errors = sum(1 for a, b in zip(test_bits[:min_length], recovered_bitstring[:min_length]) if a != b)
        ber = errors / min_length

        # Print results
        print(f"Original bits length: {len(test_bits)}")
        print(f"Recovered bits length: {len(recovered_bitstring)}")
        print(f"Compared first {min_length} bits")
        print(f"Bit errors: {errors}")
        print(f"Bit Error Rate (BER): {ber:.6f}")
    
    # test_wav_file()
    test_buffered_demodulator()
