# Apple A-chipsets: How to Estimate a Suitable Model Size for llama.cpp

## Unified Memory Architecture

On a standard PC, the CPU and GPU have separate memory pools. There may be 16GB 
worth of RAM and an NVIDIA GPU with 8GB of VRAM. The Large Language Model will 
run entirely on the GPU's 8GB VRAM. Essentially, this is the hard limit for 
model inference, meaning the 8GB GPU can only use 8GB, no matter how much RAM 
is installed.

Apple Silicon, on the other hand, uses Unified Memory Architecture (UMA), where 
the CPU and GPU share the same memory pool. On a 16GB MacBook Air, the OS 
typically uses around 3-4 GB, leaving roughly 12 GB for your model. llama.cpp 
uses Apple's Metal framework to run inference on the GPU, taking advantage of 
the full unified memory pool. This means a Mac user can effectively use more 
memory for AI than someone with a PC running a cheap GPU with only 4-8 GB of 
dedicated VRAM.

## Model Size Formula

The size of a model in memory depends on two things: the number of parameters 
and how many bytes each one uses. The formula is simple:

Model size in GB = (parameters x bytes per weight) / 1,000,000,000

For example, a 7 billion parameter model at F16 uses 2 bytes per weight, so it 
requires roughly 14 GB. The same model at Q4 uses only 0.5 bytes per weight, 
bringing it down to 3.5 GB. This is why quantization matters so much on Apple 
Silicon. It is the main way to fit larger, smarter models into your available 
memory.

## Quantization

Quantization is the process of compressing continuous or precise information 
into smaller, simpler values. In the context of AI and machine learning, it 
refers to reducing the precision of a model's weights, for example from 32-bit 
down to 4 or 8-bit. This reduction in memory usage allows larger models to run 
faster on devices with limited memory, like a MacBook Air.

- Q4 is good for most Mac users because it only takes up 3.5GB for a 7B model, leaving plenty of room for the OS.
- Q8 has better quality than Q4 but needs 7GB for a 7B model, so it works best if you have 16GB or more.
- F16 is the best quality but needs 14GB for a 7B model, so only use it if you have 32GB or more.
- Q2 is the smallest option but quality suffers noticeably, so only use it if your Mac has 8GB and nothing else fits.

## How Much Can My Mac Run?

Use your Mac's unified memory size to find the right model for you.

**8 GB Mac**
Your effective memory budget is around 4 GB after the OS takes its share.
Best options: 7B Q2 or 3B Q4

**16 GB Mac**
Your effective memory budget is around 12 GB.
Best options: 7B Q8 or 13B Q4

**24 GB Mac**
Your effective memory budget is around 20 GB.
Best options: 13B Q8 or 30B Q4

**32 GB Mac**
Your effective memory budget is around 28 GB.
Best options: 30B Q8 or 70B Q4

**64 GB Mac**
Your effective memory budget is around 60 GB.
Best options: 70B Q8

**96 GB Mac**
Your effective memory budget is around 90 GB.
Best options: 70B F16

## Practical llama.cpp Commands

Before downloading a large model, you can check how much memory it will 
need by using the following command:

./build/bin/llama-cli -m your-model.gguf -n 1 --verbose

This will show you how much memory the model is using before you commit 
to a full run. If the number is close to your available memory, consider 
downloading a more compressed version instead.

To run a model with Metal GPU acceleration on your Mac:

./build/bin/llama-cli -m your-model.gguf -ngl 99 -p "Hello"

The -ngl 99 flag tells llama.cpp to offload as many layers as possible 
to the GPU, which is what you want on Apple Silicon to take full advantage 
of unified memory.

## Recommendations by Mac Memory Size

If you have a 16GB Mac, you should be aware that the OS will occupy around 
3-4GB leaving you with 12GB. In this case, you should look for a model with 
either 7 or 13 billion parameters. The model should be downloaded with the Q8 
version as this will ensure you have the greatest quality and be able to 
comfortably fit within your remaining memory.

If you have an 8GB Mac, the OS will take around 3-4GB leaving you with about 
4GB free. Look for models with 3 or 7 billion parameters and download the Q4 
version as anything larger will not fit comfortably.

If you have a 32GB Mac, you have around 28GB available after the OS. You can 
comfortably run 30 or 70 billion parameter models at Q4 or Q8 quality.
