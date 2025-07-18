import subprocess
import re
import sys

##run this script with example python3 model-rerun.py /proj/rel/sw/ggml/models/Tiny-Llama-v0.3-FP32-1.1B-F32.gguf
# Check for model path argument
if len(sys.argv) < 2:
    print("Usage: python3 model_rerun.py /full/path/to/model.gguf")
    sys.exit(1)

model_path = sys.argv[1]

# Base paragraph to repeat
base_prompt = (
    "Use the following pieces of context to answer the question at the end. "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
    "Symbol used by Apple and Google on some devices to denote an Ethernet connection. "
    "Ethernet is a family of wired computer networking technologies used in LAN, MAN, and WAN. "
    "It was introduced in 1980 and standardized in 1983 as IEEE 802.3. "
    "Over time, Ethernet has replaced technologies like Token Ring and ARCNET. "
    "The original 10BASE5 Ethernet used a thick coaxial cable. "
    "Question: What is Ethernet? Helpful Answer: An Ethernet network is a type of computer network. "
    "Next topic: California. California, often called the 'Golden State', is the most populous U.S. state. "
    "It stretches along the Pacific Ocean and features diverse geography from beaches to mountains. "
)

# Prompt size multipliers
multipliers = [1, 2, 3, 4, 5]
results = []

#for i, multiplier in enumerate(multipliers, start=1):
for i, multiplier in enumerate(multipliers[4:], start=5):
    prompt = base_prompt * multiplier
    prompt_length = len(prompt)
    print(f"\n🔄 Run {i}: Testing with prompt size {multiplier}x, actual size = {prompt_length} characters")

    command = [
        "./build-posix/bin/llama-cli",
        "-p", prompt,
        "-m", model_path,
        "--device", "none",
        "-c", "12288",
        "--temp", "0.0",
        "--n-predict", "5",
        "--repeat-penalty", "1.5",
        "-b", "1024",
        "--top-k", "50",
        "--top-p", "0.9",
        "--repeat-last-n", "5",
        "--no-warmup"
    ]

    print("🚀 Executing llama-cli...")
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
        print("✅ Execution complete.")
    except subprocess.CalledProcessError as e:
        output = e.output
        print("⚠️ Execution failed, capturing output.")

    print("🔍 Parsing performance metrics...")
    load_time = re.search(r"load time\s*=\s*([\d.]+) ms", output)
    prompt_eval_time = re.search(r"prompt eval time\s*=\s*([\d.]+) ms", output)
    eval_time = re.search(r"eval time\s*=\s*([\d.]+) ms", output)

    results.append({
        "Run": i,
        "Prompt Size": f"{multiplier}x",
        "Load Time (ms)": float(load_time.group(1)) if load_time else "N/A",
        "Prompt Eval Time (ms)": float(prompt_eval_time.group(1)) if prompt_eval_time else "N/A",
        "Eval Time (ms)": float(eval_time.group(1)) if eval_time else "N/A"
    })

    print("📦 Metrics captured.")

# Final summary
print("\n📊 Benchmark Summary:")
print(f"{'Run':<5} {'Prompt Size':<12} {'Load Time (ms)':<18} {'Prompt Eval Time (ms)':<24} {'Eval Time (ms)':<18}")
for result in results:
    print(f"{result['Run']:<5} {result['Prompt Size']:<12} {str(result['Load Time (ms)']):<18} {str(result['Prompt Eval Time (ms)']):<24} {str(result['Eval Time (ms)']):<18}")

