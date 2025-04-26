"""
For now, just open the microphone and listen for the sound then put it into wav.

AudioSource -> SignalDetector -> Demodulator -> PacketDecoder
      |            |                |               |
      └────────────┴────────────────┴───────────────┘
                              |
                           Receiver
AudioSource: The microphone that receives the audio signal, but later we can use phone call connections.
SignalDetector: The signal detector that detects the signal and converts it into a digital signal.
Demodulator: The demodulator that demodulates the signal and converts it into a bitstream.
PacketDecoder: The packet decoder that decodes the bitstream into packets.
The receiver will receive the audio signal from the sender and decode it into a bitstream.
The receiver will then decode the bitstream into packets and send them to the receiver.
The receiver will then decode the packets and send them to the receiver.
"""

import numpy as np
import pyaudio
import wave
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, BinaryIO


class AudioSource:
    """
    Captures audio input from a microphone.
    In the future, this could be extended to capture from phone call connections.
    """
    def __init__(self, sample_rate: int = 44100, chunk_size: int = 1024, channels: int = 1):
        """
        Initialize the audio source.
        
        Args:
            sample_rate: Sampling rate in Hz
            chunk_size: Number of frames per buffer
            channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paInt16
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.recorded_frames = []
    
    def start_recording(self):
        """Start recording audio from the microphone."""
        if self.is_recording:
            return
            
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        self.is_recording = True
        self.recorded_frames = []
        
    def read_chunk(self) -> bytes:
        """
        Read a chunk of audio data from the microphone.
        
        Returns:
            Raw audio data as bytes
        """
        if not self.is_recording:
            self.start_recording()
            
        data = self.stream.read(self.chunk_size)
        self.recorded_frames.append(data)
        return data
    
    def stop_recording(self):
        """Stop recording audio."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    
    def save_to_wav(self, filename: str):
        """
        Save recorded audio to a WAV file.
        
        Args:
            filename: Path to save the WAV file
        """
        if not self.recorded_frames:
            raise ValueError("No audio data has been recorded")
            
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.recorded_frames))
        wf.close()
        
    def close(self):
        """Clean up resources."""
        self.stop_recording()
        self.audio.terminate()


class SignalDetector:
    """
    Detects audio signals and converts them into digital signals.
    """
    def __init__(self, threshold: float = 0.1, window_size: int = 10):
        """
        Initialize the signal detector.
        
        Args:
            threshold: Energy threshold for signal detection
            window_size: Size of the detection window in chunks
        """
        self.threshold = threshold
        self.window_size = window_size
        self.energy_history = []
        
    def detect_signal(self, audio_chunk: bytes) -> Tuple[bool, np.ndarray]:
        """
        Detect if the audio chunk contains a signal.
        
        Args:
            audio_chunk: Raw audio data as bytes
            
        Returns:
            Tuple of (signal_detected, digital_signal)
        """
        # Convert audio chunk to numpy array
        signal = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # Calculate signal energy
        energy = np.mean(np.abs(signal))
        self.energy_history.append(energy)
        
        # Keep only the latest window_size energy values
        if len(self.energy_history) > self.window_size:
            self.energy_history.pop(0)
        
        # Signal is detected if the mean energy is above the threshold
        signal_detected = np.mean(self.energy_history) > self.threshold
        
        return signal_detected, signal
    
    def reset(self):
        """Reset the detector state."""
        self.energy_history = []


class Demodulator:
    """
    Demodulates the digital signal into a bitstream.
    """
    def __init__(self, sample_rate: int = 44100, bit_rate: int = 1200):
        """
        Initialize the demodulator.
        
        Args:
            sample_rate: Sampling rate in Hz
            bit_rate: Bit rate in bits per second
        """
        self.sample_rate = sample_rate
        self.bit_rate = bit_rate
        self.samples_per_bit = sample_rate // bit_rate
        self.buffer = np.array([], dtype=np.int16)
        
    def demodulate(self, digital_signal: np.ndarray) -> List[int]:
        """
        Demodulate the digital signal into bits.
        This is a simple implementation that assumes amplitude-shift keying (ASK).
        
        Args:
            digital_signal: Digital signal as numpy array
            
        Returns:
            List of demodulated bits (0 or 1)
        """
        # Append new signal to buffer
        self.buffer = np.append(self.buffer, digital_signal)
        
        # Extract as many complete bits as possible
        num_complete_bits = len(self.buffer) // self.samples_per_bit
        bits = []
        
        for i in range(num_complete_bits):
            # Extract the samples for this bit
            bit_samples = self.buffer[i * self.samples_per_bit:(i + 1) * self.samples_per_bit]
            
            # Simple threshold-based demodulation (ASK)
            # If the average amplitude is above half the max possible amplitude, it's a 1
            bit_value = 1 if np.mean(np.abs(bit_samples)) > 16384/2 else 0
            bits.append(bit_value)
        
        # Keep remaining samples in buffer
        self.buffer = self.buffer[num_complete_bits * self.samples_per_bit:]
        
        return bits
    
    def reset(self):
        """Reset the demodulator state."""
        self.buffer = np.array([], dtype=np.int16)


class PacketDecoder:
    """
    Decodes bitstreams into packets.
    """
    def __init__(self, packet_size: int = 8, header_pattern: List[int] = None):
        """
        Initialize the packet decoder.
        
        Args:
            packet_size: Size of a packet in bits
            header_pattern: Pattern that indicates the start of a packet
        """
        self.packet_size = packet_size
        self.header_pattern = header_pattern if header_pattern else [1, 1, 0, 1, 0, 1, 0, 1]  # Default header
        self.header_length = len(self.header_pattern)
        self.bit_buffer = []
        self.packets = []
        
    def decode(self, bits: List[int]) -> List[List[int]]:
        """
        Decode bits into packets.
        
        Args:
            bits: List of bits (0 or 1)
            
        Returns:
            List of decoded packets
        """
        # Add new bits to buffer
        self.bit_buffer.extend(bits)
        
        # Process as many packets as possible
        decoded_packets = []
        
        # Keep processing while we might have a complete packet in the buffer
        while len(self.bit_buffer) >= self.header_length + self.packet_size:
            # Look for header pattern
            for i in range(len(self.bit_buffer) - self.header_length + 1):
                if self.bit_buffer[i:i+self.header_length] == self.header_pattern:
                    # Found header, try to extract packet
                    start_idx = i + self.header_length
                    end_idx = start_idx + self.packet_size
                    
                    # Make sure we have enough bits for a full packet
                    if end_idx <= len(self.bit_buffer):
                        packet = self.bit_buffer[start_idx:end_idx]
                        decoded_packets.append(packet)
                        
                        # Remove bits up to the end of this packet
                        self.bit_buffer = self.bit_buffer[end_idx:]
                        break
            else:
                # No complete packet found, keep only the last (header_length - 1) bits
                # as they might be part of a header that spans across chunks
                if len(self.bit_buffer) > self.header_length:
                    self.bit_buffer = self.bit_buffer[-(self.header_length - 1):]
                break
                
        return decoded_packets
    
    def reset(self):
        """Reset the decoder state."""
        self.bit_buffer = []
        self.packets = []


class Receiver:
    """
    Combines all components to receive and decode audio signals.
    """
    def __init__(self, output_file: str = "received_audio.wav"):
        """
        Initialize the receiver with all necessary components.
        
        Args:
            output_file: Path to save the recorded audio
        """
        self.audio_source = AudioSource()
        self.signal_detector = SignalDetector()
        self.demodulator = Demodulator()
        self.packet_decoder = PacketDecoder()
        self.output_file = output_file
        self.is_running = False
        self.decoded_packets = []
        
    def start(self, duration: Optional[float] = None):
        """
        Start receiving and processing audio.
        
        Args:
            duration: Optional duration in seconds to record. If None, runs until stop() is called.
        """
        self.is_running = True
        self.audio_source.start_recording()
        
        start_time = time.time()
        
        try:
            while self.is_running:
                # Check if we've reached the requested duration
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Get audio chunk
                audio_chunk = self.audio_source.read_chunk()
                
                # Detect signal
                signal_detected, digital_signal = self.signal_detector.detect_signal(audio_chunk)
                
                if signal_detected:
                    # Demodulate signal to bits
                    bits = self.demodulator.demodulate(digital_signal)
                    
                    # Decode bits to packets
                    packets = self.packet_decoder.decode(bits)
                    
                    # Store decoded packets
                    if packets:
                        self.decoded_packets.extend(packets)
                        print(f"Decoded {len(packets)} packets")
                
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stop receiving and save the recorded audio."""
        self.is_running = False
        self.audio_source.stop_recording()
        self.audio_source.save_to_wav(self.output_file)
        print(f"Recording saved to {self.output_file}")
        print(f"Total decoded packets: {len(self.decoded_packets)}")
        
    def reset(self):
        """Reset the state of all components."""
        self.signal_detector.reset()
        self.demodulator.reset()
        self.packet_decoder.reset()
        self.decoded_packets = []
        
    def close(self):
        """Clean up resources."""
        self.audio_source.close()


# Example usage
if __name__ == "__main__":
    # Create a receiver
    receiver = Receiver("received_signal.wav")
    
    try:
        print("Listening for audio signals. Press Ctrl+C to stop...")
        # Start receiving for 10 seconds
        receiver.start(duration=10)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        receiver.close()
        print("Done!")