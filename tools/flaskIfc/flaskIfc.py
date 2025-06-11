from flask import Flask, request
import subprocess

app = Flask(__name__)

@app.route('/serial', methods=['GET'])
def serial_command():
    # Currently the port is hard coded to /dev/ttyUSB3 but can be parameterized
    port = '/dev/ttyUSB3'
    #port = request.args.get('port')

    # Currently the baudrate is hard coded to 921600 but can be parameterized
    #baudrate = request.args.get('baudrate')
    baudrate = '921600'


    # Parse the command and send it to serial.py 
    command = request.args.get('command')

    #if not all([port, baudrate, command]):
    if not all([command]):
        return "Missing parameters", 400

    try:
        result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
        return result.stdout.strip(), 200
    except subprocess.CalledProcessError as e:
        return f"Error executing script: {e.stderr}", 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
