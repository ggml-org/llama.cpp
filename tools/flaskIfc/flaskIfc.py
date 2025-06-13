from flask import Flask, render_template, request
import subprocess
import threading
import time

job_status = {"running": False, "result": "", "thread": None}

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/llama-cli', methods=['GET'])
def serial_command():
    # Currently the port is hard coded to /dev/ttyUSB3 but can be parameterized
    port = '/dev/ttyUSB3'
    #port = request.args.get('port')

    # Currently the baudrate is hard coded to 921600 but can be parameterized
    #baudrate = request.args.get('baudrate')
    baudrate = '921600'
    #./run_platform_test.sh "my cat's name" "10" "tinyllama-vo-5m-para.gguf" "none"
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
    script_path = "/usr/bin/tsi/v0.1.1.tsv31_06_06_2025/bin/run_platform_test.sh"
    command = f"{script_path} \"{prompt}\" {tokens} {model_path} {backend}"

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

    #./run_platform_test.sh "my cat's name" "10" "tinyllama-vo-5m-para.gguf" "none"
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
    # Currently the port is hard coded to /dev/ttyUSB3 but can be parameterized
    port = '/dev/ttyUSB3'

    # Currently the baudrate is hard coded to 921600 but can be parameterized
    baudrate = '921600'
    script_path = "/usr/bin/tsi/v0.1.1.tsv31_06_06_2025/bin/run_platform_test.sh"
    command = f"{script_path} \"{prompt}\" {tokens} {model_path} {backend}"


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
