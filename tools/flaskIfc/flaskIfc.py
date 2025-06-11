from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
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

   # Below is for reference i will remove later
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


    try:
        result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
        output = result.stdout  # This should have \n
    except subprocess.CalledProcessError as e:
        output = f"Error running model: {e.stderr}"

    return render_template('result.html', output=output)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
