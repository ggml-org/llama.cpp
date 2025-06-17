from flask import Flask, render_template, request
import subprocess
import threading
import time
from werkzeug.utils import secure_filename
import os

job_status = {"running": False, "result": "", "thread": None}

app = Flask(__name__)

port = '/dev/ttyUSB3'
baudrate = '921600'
exe_path = "/usr/bin/tsi/v0.1.1.tsv31_06_06_2025/bin/"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/llama-cli', methods=['GET'])
def llama_cli_serial_command():

    #./run_llama_cli.sh "my cat's name" "10" "tinyllama-vo-5m-para.gguf" "none"
    model = request.args.get('model')
    backend = request.args.get('backend')
    tokens = request.args.get('tokens')
    prompt = request.args.get('prompt')

    # Define the model path (update with actual paths)
    model_paths = {
        "tiny-llama": "tinyllama-vo-5m-para.gguf",
        "Tiny-llama-F32": "Tiny-Llama-v0.3-FP32-1.1B-F32.gguf"
    }

    model_path = model_paths.get(model, "")
    if not model_path:
        return f"<h2>Error: Model path not found for '{model}'</h2>"

    # Build llama-cli command
    #command = [
    #    "./llama-cli",
    #    "-p", prompt,
    #    "-m", model_path,
    #    "--device", backend,
    #    "--temp", "0",
    #    "--n-predict", tokens,
    #    "--repeat-penalty", "1",
    #    "--top-k", "0",
    #    "--top-p", "1"
    #]
    # URL to Test this end point is as follows
    # http://10.50.30.167:5001/llama-cli?model=tiny-llama&backend=tSavorite&tokens=5&prompt=Hello+How+are+you
    script_path = "./run_llama_cli.sh"
    command = f"cd {exe_path}; {script_path} \"{prompt}\" {tokens} {model_path} {backend}"

    try:
        result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
        return result.stdout, 200
    except subprocess.CalledProcessError as e:
        return f"Error executing script: {e.stderr}", 500

UPLOAD_FOLDER = './' # Directory where uploaded files will be stored
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Create the upload folder if it doesn't exist

@app.route('/upload-gguf', methods=['POST', 'GET'])
def upload_serial_command():
    if request.method == 'POST':
        # Check if a file was submitted
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return "No file selected"

       # Save the file if it exists
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return "File uploaded successfully"
    return render_template('upload.html') # Display the upload form

#    command = f"upload file"
#    try:
#        result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
#        return result.stdout, 200
#    except subprocess.CalledProcessError as e:
#        return f"Error executing script: {e.stderr}", 500

@app.route('/upload-file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file was submitted
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return "No file selected"

        # Save the file if it exists
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return "File uploaded successfully"
    return render_template('upload.html') # Display the upload form

@app.route('/restart-txe', methods=['GET'])
def restart_txe_serial_command():
    command = f"telnet localhost 8000\r\nclose all\r\n"

    try:
        result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
        time.sleep(5)
        command = f"{exe_path}/../install/tsi-start\nyes\n"
        try:
            result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
            return result.stdout, 200
        except subprocess.CalledProcessError as e:
            return f"Error executing script: {e.stderr}", 500
    except subprocess.CalledProcessError as e:
        return f"Error executing script: {e.stderr}", 500

@app.route('/health-check', methods=['GET'])
def health_check_serial_command():
    command = f"free -h"

    try:
        result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
        return result.stdout, 200
    except subprocess.CalledProcessError as e:
        return f"Error executing script: {e.stderr}", 500

@app.route('/test', methods=['GET'])
def test_serial_command():
    command = f"test"

    try:
        result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
        return result.stdout, 200
    except subprocess.CalledProcessError as e:
        return f"Error executing script: {e.stderr}", 500

@app.route('/system-info', methods=['GET'])
def system_info_serial_command():

    command = f"{exe_path}../install/tsi-version;lscpu"

    try:
        result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
        return result.stdout, 200
    except subprocess.CalledProcessError as e:
        return f"Error executing script: {e.stderr}", 500

@app.route('/submit', methods=['POST'])
def submit():
    global job_status

    if job_status["running"]:
        return "<h2>A model is already running. Please wait or abort.</h2>"

    #./run_llama_cli.sh "my cat's name" "10" "tinyllama-vo-5m-para.gguf" "none"
    model = request.form.get('model')
    backend = request.form.get('backend')
    tokens = request.form.get('tokens')
    prompt = request.form.get('prompt')

    # Define the model path (update with actual paths)
    model_paths = {
        "tiny-llama": "tinyllama-vo-5m-para.gguf",
        "Tiny-llama-F32": "Tiny-Llama-v0.3-FP32-1.1B-F32.gguf"
    }

    model_path = model_paths.get(model, "")
    if not model_path:
        return f"<h2>Error: Model path not found for '{model}'</h2>"

    # Build llama-cli command
    #command = [
    #    "./llama-cli",
    #    "-p", prompt,
    #    "-m", model_path,
    #    "--device", backend,
    #    "--temp", "0",
    #    "--n-predict", tokens,
    #    "--repeat-penalty", "1",
    #    "--top-k", "0",
    #    "--top-p", "1"
    #]

    script_path = "./run_llama_cli.sh"
    command = f"cd {exe_path}; {script_path} \"{prompt}\" {tokens} {model_path} {backend}"


    def run_script():
        try:
            result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
            job_status["result"] = result.stdout
        except subprocess.CalledProcessError as e:
            job_status["result"] = f"Error: {e.stderr}"
        finally:
            job_status["running"] = False

    thread = threading.Thread(target=run_script)
    job_status = {"running": True, "result": "", "thread": thread}
    thread.start()

    return render_template("processing.html")

@app.route('/status')
def status():
    if job_status["running"]:
        return "running"
    else:
        return "done"

@app.route('/result')
def result():
    return render_template("result.html", output=job_status["result"])

@app.route('/abort')
def abort():
    global job_status
    if job_status["running"] and job_status["thread"].is_alive():
        # Use subprocess.Popen + pid handling instead for real process termination
        job_status["running"] = False
        job_status["result"] = "Aborted by user."
        return "<h2>Job aborted.</h2><a href='/'>Home</a>"
    return "<h2>No job running.</h2><a href='/'>Home</a>"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
