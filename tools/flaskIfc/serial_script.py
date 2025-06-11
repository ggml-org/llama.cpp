import serial
import sys

def send_serial_command(port, baudrate, command):
    try:
        # Open the serial port with 1 second timeout
        ser = serial.Serial(port, baudrate, timeout=20)

        ser.write(command.encode())  # Encode command to bytes
        ser.write('\n'.encode())  # Encode command to bytes
        
        # Wait to read the serial port
        data = '\0'
        while True:
            try:
                line = ser.readline()
                if line: # Check if line is not empty
                    data += line.decode('utf-8')  # Keep the line as-is with newline
                else:
                    break  # Exit loop if no data is received
            except serial.SerialException as e:
                ser.close()
                return (f"Error reading from serial port: {e}")
            except KeyboardInterrupt:
                ser.close()
                return ("Program interrupted by user")
        ser.close()
        return data

    except serial.SerialException as e:
        ser.close()
        return f"Error: {e}"

# This script can be run in standalone as well
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <port> <baudrate> <command>")
        sys.exit(1)

    port = sys.argv[1]
    baudrate = int(sys.argv[2])
    command = sys.argv[3]
    response = send_serial_command(port, baudrate, command)
