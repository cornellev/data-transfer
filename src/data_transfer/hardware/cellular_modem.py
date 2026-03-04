import RPi.GPIO as GPIO
import serial
import time

class CellularModem:
    def __init__(self, power_key=6, port="/dev/ttyS0", baud=115200):
        """
        Initialize the modem.
        - power_key: GPIO pin connected to SIM7600 PWRKEY
        - port: serial port for AT commands
        - baud: UART baud rate (default: 115200)
        """
        self.power_key = power_key
        self.ser = serial.Serial(port, baud)
        self.ser.flushInput()

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.power_key, GPIO.OUT)

    def power_on(self):
        """
        Turn on the SIM7600.
        """
        GPIO.output(self.power_key, GPIO.HIGH)
        time.sleep(2)
        GPIO.output(self.power_key, GPIO.LOW)
        time.sleep(20)  
        self.ser.flushInput()
        print("SIM7600 is powered on.")

    def power_down(self):
        """
        Turn off the SIM7600.
        """
        GPIO.output(self.power_key, GPIO.HIGH)
        time.sleep(3)
        GPIO.output(self.power_key, GPIO.LOW)
        time.sleep(18)
        print("SIM7600 is powered off.")
    
    def send_at(self, command, back, timeout=1):
        """
        Send an AT command and check for the expected response.
        """
        self.ser.write((command + '\r\n').encode())
        time.sleep(timeout)

        resp = ""
        if self.ser.inWaiting():
            resp = self.ser.read(self.ser.inWaiting()).decode(errors="ignore")

        if back not in resp:
            print(f"{command} ERROR: {resp.strip()}")
            return False
        
        print(resp.strip())
        return True

    def dial(self, number):
        """
        Dial a number (voice call).
        """
        return self.send_at(f"ATD{number};", "OK", 2)

    def answer_call(self):
        """
        Answer an incoming call.
        """
        return self.send_at("ATA", "OK", 2)

    def hangup(self):
        """
        Hang up an active call.
        """
        self.ser.write(b"AT+CHUP\r\n")
        time.sleep(1)
        print("Call disconnected.")

    def close(self):
        """
        Cleanup resources (serial and GPIO).
        """
        if self.ser:
            self.ser.close()
        GPIO.cleanup()
