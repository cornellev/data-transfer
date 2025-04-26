"""
For now, just open the microphone and listen for the sound then stream that into the BufferedDecoder.
This is a placeholder for the actual implementation of the receiver that uses the telephone protocol.

Receiver class would just be a wrapper around the actual receiver implementation, which is another class called LaptopMicrophone.
Later it will be implemented as a telephone voice call.
"""

from .audio_decoder import BufferedDecoder



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