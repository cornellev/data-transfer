"""
For now, just open the microphone and listen for the sound then stream that into the BufferedDecoder.
This is a placeholder for the actual implementation of the receiver that uses the telephone protocol.

Receiver class would just be a wrapper around the actual receiver implementation, which is another class called LaptopMicrophone.
Later it will be implemented as a telephone voice call.
"""

from .audio_decoder import BufferedDemodulator

import pyaudio
import numpy as np
import threading
import time
from typing import Optional, Tuple
from scipy import signal

class Receiver:
    """
    Audio receiver that captures microphone input and decodes 16-QAM signals in real-time.
    """
    def __init__(self, carrier_freq=4000, sample_rate=44100, symbol_rate=500, 
                 chunk_size=1024, channels=1):
        """
        Initialize the audio receiver.
        
        Args:
            carrier_freq: Carrier frequency in Hz
            sample_rate: Sampling rate in Hz
            symbol_rate: Symbol rate in symbols per second
            chunk_size: Number of samples per chunk
            channels: Number of audio channels (1 for mono)
        """
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.chunk_size = chunk_size
        self.channels = channels
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Initialize the demodulator
        self.demodulator = BufferedDemodulator(
            carrier_freq=carrier_freq,
            sample_rate=sample_rate,
            symbol_rate=symbol_rate
        )
        
        # Thread control
        self.running = False
        self.receive_thread = None
        
        # Buffer for storing decoded bits
        self.bit_buffer = []
        self.bit_buffer_lock = threading.Lock()
    
    def _process_chunk(self, audio_data):
        """
        Process a chunk of audio data.
        """
        # Demodulate the signal
        bits = self.demodulator.demodulate(audio_data)
        
        if bits and len(bits) > 0:
            # Acquire lock before modifying the shared bit_buffer
            with self.bit_buffer_lock:
                self.bit_buffer.extend(bits)
                
                # Print the received bits in chunks of 8 (bytes)
                self._process_bit_buffer()
    
    def _process_bit_buffer(self):
        """
        Process received bits by decoding them into characters based on 8-bit ASCII encoding.
        """
        decoded_message = ""
        
        # Process bits in chunks of 8 (one byte/character)
        while len(self.bit_buffer) >= 8:
            # Take 8 bits at a time
            byte_bits = self.bit_buffer[:8]
            self.bit_buffer = self.bit_buffer[8:]
            
            # Convert binary to decimal (ASCII value)
            byte_value = int(''.join(map(str, byte_bits)), 2)
            
            # Convert to character and add to decoded message
            if 32 <= byte_value <= 126:  # Printable ASCII range
                char = chr(byte_value)
                decoded_message += char
                print(char, end='', flush=True)
            else:
                # Handle special characters (newlines, etc.)
                if byte_value == 10:  # Newline
                    decoded_message += "\n"
                    print("\\n", end='', flush=True)
                elif byte_value == 13:  # Carriage return
                    decoded_message += "\r"
                    print("\\r", end='', flush=True)
                elif byte_value == 9:  # Tab
                    decoded_message += "\t"
                    print("\\t", end='', flush=True)
                else:
                    # For other non-printable characters, just show hex
                    print(f"\\x{byte_value:02x}", end='', flush=True)
        
        # Save decoded message to file if we have anything
        if decoded_message:
            with open("decoded_message.txt", "a") as f:
                f.write(decoded_message)
    
    def _receive_loop(self, duration: Optional[float] = None):
        """
        Main reception loop that runs in a separate thread.
        
        Args:
            duration: Maximum duration in seconds (None for infinite)
        """
        start_time = time.time()
        
        try:
            # Open stream for input (non-callback mode)
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            print("Receiver started - listening for QAM signals...")
            
            # Continuously read audio chunks while running
            while self.running:
                # Read audio data
                try:
                    audio_data = self.stream.read(self.chunk_size)
                    # Convert buffer to numpy array
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    # Process the data
                    self._process_chunk(audio_np)
                except Exception as e:
                    print(f"Error reading audio: {e}")
                
                # Check if we've reached the duration limit
                if duration is not None and time.time() - start_time > duration:
                    self.running = False
                    break
                
        except Exception as e:
            print(f"Error in receive loop: {e}")
        finally:
            # Clean up
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            print("\nReceiver stopped")
    
    def start(self, duration: Optional[float] = None):
        """
        Start listening for audio signals.
        
        Args:
            duration: Maximum duration in seconds (None for infinite)
        """
        if self.running:
            print("Receiver is already running")
            return
        
        # Reset state
        self.running = True
        self.demodulator.reset()
        with self.bit_buffer_lock:
            self.bit_buffer = []
        
        # Start the receive thread
        self.receive_thread = threading.Thread(target=self._receive_loop, args=(duration,))
        self.receive_thread.daemon = True
        self.receive_thread.start()
        
        # If duration is specified, wait for completion
        if duration is not None:
            # Wait until duration expires
            time.sleep(duration)
            self.stop()
    
    def stop(self):
        """Stop listening for audio signals."""
        self.running = False
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2.0)
    
    def close(self):
        """Close the receiver and release resources."""
        self.stop()
        if self.audio:
            self.audio.terminate()
            self.audio = None


from typing import Tuple
import numpy as np
from scipy import signal

class SignalDetector:
    """
    Adaptive signal detector for QAM signals.
    Supports both Butterworth-based and FFT-based bandpass filtering.
    """
    def __init__(self, alpha=0.01, initial_threshold=0.01, use_fft=False):
        """
        Args:
            alpha: Smoothing factor for updating noise floor
            initial_threshold: Initial energy threshold estimate
            use_fft: Whether to use FFT-based filtering instead of Butterworth
        """
        self.noise_floor = initial_threshold
        self.alpha = alpha
        self.use_fft = use_fft

    def detect(
        self,
        audio_data: np.ndarray,
        fs: float,
        fc: float,
        bandwidth: float = 1000,
        threshold_factor: float = 3.0
    ) -> Tuple[bool, np.ndarray]:
        """
        Detect presence of a signal and return whether signal is present and the filtered data.
        """
        if self.use_fft:
            filtered = self._fft_bandpass_filter(audio_data, fs, fc, bandwidth)
        else:
            filtered = self._butter_bandpass_filter(audio_data, fs, fc, bandwidth)

        # Calculate signal energy
        energy = np.mean(filtered**2)

        # Update noise floor estimate if likely no signal present
        if energy < self.noise_floor * threshold_factor:
            self.noise_floor = (1 - self.alpha) * self.noise_floor + self.alpha * energy

        signal_present = energy > self.noise_floor * threshold_factor
        return signal_present, filtered

    def _butter_bandpass_filter(self, audio_data, fs, fc, bandwidth=1000) -> np.ndarray:
        """
        Apply a 5th-order Butterworth bandpass filter.
        """
        nyquist = fs / 2
        low = max(0.001, (fc - bandwidth / 2) / nyquist)
        high = min(0.999, (fc + bandwidth / 2) / nyquist)
        b, a = signal.butter(5, [low, high], btype='band')
        return signal.filtfilt(b, a, audio_data)

    def _fft_bandpass_filter(self, audio_data, fs, fc, bandwidth=1000) -> np.ndarray:
        """
        Apply bandpass filtering using FFT.
        """
        N = len(audio_data)
        freq_data = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(N, d=1/fs)

        # Build mask for desired frequency band
        mask = np.logical_and(np.abs(freqs) >= (fc - bandwidth / 2),
                              np.abs(freqs) <= (fc + bandwidth / 2))
        freq_data[~mask] = 0  # Zero out unwanted frequencies

        # Convert back to time domain
        filtered = np.fft.ifft(freq_data).real
        return filtered

class EnhancedReceiver:
    """
    Enhanced audio receiver with noise reduction capabilities that wraps around
    the base Receiver class.
    """
    def __init__(self, receiver, detection_method='adaptive', 
                 threshold_factor=3.0, show_snr=True):
        """
        Initialize the enhanced receiver with a base receiver.
        
        Args:
            receiver: An instance of the Receiver class
            detection_method: One of 'threshold', 'adaptive', or 'correlation'
            threshold_factor: Factor applied to noise floor for detection threshold
            show_snr: Whether to display SNR information
        """
        self.receiver = receiver
        self.detection_method = detection_method
        self.threshold_factor = threshold_factor
        self.show_snr = show_snr
        
        # Initialize signal detector
        self.signal_detector = SignalDetector(use_fft=True)
        
        # For calculating SNR
        self.signal_power = 0
        self.noise_power = 0.001  # Small non-zero value to avoid division by zero
        self.snr_alpha = 0.1  # Smoothing factor
        
        # Save original process_chunk method and override it
        self.original_process_chunk = receiver._process_chunk
        receiver._process_chunk = self._enhanced_process_chunk
        
        # Status tracking
        self.signal_detected = False
        self.last_signal_time = 0
        self.continuous_silence_time = 0
    
    def _enhanced_process_chunk(self, audio_data):
        """
        Enhanced processing of audio chunks with denoising.
        This overrides the receiver's _process_chunk method.
        """
        # Apply signal detection and filtering based on method
        if self.detection_method == 'adaptive':
            signal_present, filtered_data = self.signal_detector.detect(
                audio_data, 
                self.receiver.sample_rate, 
                self.receiver.carrier_freq,
                bandwidth=self.receiver.symbol_rate * 2,
                threshold_factor=self.threshold_factor
            )
        elif self.detection_method == 'correlation':
            signal_present, filtered_data = self._detect_qam_signal(
                audio_data, 
                self.receiver.sample_rate, 
                self.receiver.carrier_freq
            )
        else:  # fallback to threshold
            signal_present, filtered_data = self._detect_signal(
                audio_data, 
                self.receiver.sample_rate, 
                self.receiver.carrier_freq
            )
        
        current_time = time.time()
        
        # Update signal and noise power estimates for SNR calculation
        if signal_present:
            # Reset continuous silence counter
            self.continuous_silence_time = 0
            
            # Update signal status
            if not self.signal_detected:
                print("\nSignal detected!")
                self.signal_detected = True
            
            self.last_signal_time = current_time
            
            # Update signal power estimate
            current_power = np.mean(filtered_data**2)
            self.signal_power = (1 - self.snr_alpha) * self.signal_power + self.snr_alpha * current_power
            
            # Calculate and display SNR
            if self.show_snr:
                snr_db = 10 * np.log10(self.signal_power / self.noise_power) if self.noise_power > 0 else 0
                print(f"SNR: {snr_db:.1f} dB", end="\r")
            
            # Process with the demodulator
            bits = self.receiver.demodulator.demodulate(filtered_data)
            
            if bits and len(bits) > 0:
                with self.receiver.bit_buffer_lock:
                    self.receiver.bit_buffer.extend(bits)
                    self.receiver._process_bit_buffer()
        else:
            # Update noise power estimate when no signal present
            current_power = np.mean(audio_data**2)
            self.noise_power = (1 - self.snr_alpha) * self.noise_power + self.snr_alpha * current_power
            
            # Update signal loss status after a short delay
            if self.signal_detected and (current_time - self.last_signal_time) > 1.0:
                print("\nSignal lost. Listening...")
                self.signal_detected = False
            
            # Update continuous silence time
            if not self.signal_detected:
                self.continuous_silence_time += len(audio_data) / self.receiver.sample_rate
                
                # Print listening status periodically
                if int(self.continuous_silence_time) % 5 == 0 and abs(self.continuous_silence_time % 5) < 0.1:
                    print(f"Listening... ({int(self.continuous_silence_time)}s)", end="\r")
    
    def _detect_signal(self, audio_data, fs, fc, bandwidth=1000, threshold=0.01) -> Tuple[bool, np.ndarray]:
        """
        Detect if a QAM signal is present based on energy in the band.
        """
        # Apply bandpass filter
        filtered = self._apply_bandpass_filter(audio_data, fs, fc, bandwidth)
        
        # Calculate signal energy
        energy = np.mean(filtered**2)
        
        # Compare with threshold
        return energy > threshold, filtered
    
    def _apply_bandpass_filter(self, audio_data, fs, fc, bandwidth=1000) -> np.ndarray:
        """
        Apply a bandpass filter around the carrier frequency.
        """
        # Calculate filter parameters
        nyquist = fs / 2
        low = max(0.001, (fc - bandwidth/2) / nyquist)
        high = min(0.999, (fc + bandwidth/2) / nyquist)
        
        # Create bandpass filter
        b, a = signal.butter(5, [low, high], btype='band')
        
        # Apply filter
        filtered_data = signal.filtfilt(b, a, audio_data)
        
        return filtered_data
    
    def _detect_qam_signal(self, audio_data, fs, fc, detection_window=0.1) -> Tuple[bool, np.ndarray]:
        """
        Detect QAM signal using correlation with carrier.
        """
        # Create time base
        samples = len(audio_data)
        t = np.arange(samples) / fs
        
        # Generate carrier signals
        i_carrier = np.cos(2 * np.pi * fc * t)
        q_carrier = -np.sin(2 * np.pi * fc * t)
        
        # Correlate with carriers
        i_corr = np.abs(np.correlate(audio_data, i_carrier, mode='valid'))
        q_corr = np.abs(np.correlate(audio_data, q_carrier, mode='valid'))
        
        # Combine correlations
        total_corr = i_corr + q_corr
        
        # Normalize
        if np.max(total_corr) > 0:
            total_corr = total_corr / np.max(total_corr)
        
        # Apply bandpass filter for return
        filtered_data = self._apply_bandpass_filter(audio_data, fs, fc)
        
        # Use threshold to detect signal
        threshold = 0.5  # Adjust as needed
        signal_present = np.max(total_corr) > threshold
        
        return signal_present, filtered_data
    
    def start(self, duration: Optional[float] = None):
        """Start the receiver with the specified duration."""
        self.signal_detected = False
        self.last_signal_time = 0
        self.continuous_silence_time = 0
        self.receiver.start(duration)
    
    def stop(self):
        """Stop the receiver."""
        self.receiver.stop()
    
    def close(self):
        """Close the receiver and release resources."""
        self.receiver.close()

# Example usage
if __name__ == "__main__":
    # Create a base receiver instance
    base_receiver = Receiver(
        carrier_freq=4000, 
        sample_rate=44100, 
        symbol_rate=100,
        chunk_size=1024
    )

    # Wrap it with the enhanced receiver
    enhanced_receiver = EnhancedReceiver(
        receiver=base_receiver,
        detection_method='adaptive',  # Options: 'adaptive', 'correlation', 'threshold'
        threshold_factor=100.0,  # Adjust sensitivity (higher = less sensitive)
        show_snr=True  # Display Signal-to-Noise Ratio
    )

    # Use the enhanced receiver
    try:
        print("Listening for QAM signals with noise reduction. Press Ctrl+C to stop...")
        enhanced_receiver.start()  # Start listening indefinitely
        
        # Keep the main thread alive
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        enhanced_receiver.close()
        print("Done!")