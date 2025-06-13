import serial
import sys

def send_serial_command(port, baudrate, command):
    try:
        ser = serial.Serial(port, baudrate)

        ser.write((command + '\n').encode())  # Send command with newline
        # Wait to read the serial port
        data = '\0'
        while True:
            try:
                # read byte by byte to find either a new line character or a prompt marker
                # instead of new line using line = ser.readline()
                line = b""
                while True:
                    byte = ser.read(1)  # Read one byte at a time
                    if (byte == b"\n") or (byte == b"#"):  # Stop when delimiter is found
                        break
                    line += byte
                if line: # Check if line is not empty
                    read_next_line = line.decode('utf-8')
                    if ("run-platform-done" in read_next_line.strip()) or ("@agilex7_dk_si_agf014ea" in read_next_line.strip()) or ("imx8mpevk" in read_next_line.strip()):
                        break
                    data += read_next_line  # Keep the line as-is with newline
                else:
                    break  # Exit loop if no data is received
            except serial.SerialException as e:
                ser.close()
                return (f"Error reading from serial port: {e}")
            except KeyboardInterrupt:
                ser.close()
                return ("Program interrupted by user")
        ser.close()
        print(data)
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
