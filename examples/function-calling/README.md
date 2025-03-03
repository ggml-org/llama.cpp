# llama.cpp/examples/function-calling

This example shows how to do basic function calling using llama-cli and a python wrapper to declare and call functions.

## Options

Important options for llama-cli-function-runner.py:

-   `-m FNAME, --model FNAME`: Specify the path to the function calling model (e.g., `-m "$(huggingface-cli  download meetkai/functionary-small-v3.2-GGUF functionary-small-v3.2.Q4_0.gguf)"`).
-   `--ctx-size N`: Set the size of the prompt context. The default is 1024
-   `--special`: show special tokens and function calling details

## Example showing showing function call details

```
./examples/function-calling/llama-cli-function-runner.py -m `huggingface-cli  download meetkai/functionary-small-v3.2-GGUF functionary-small-v3.2.Q4_0.gguf` -i --special
What is the weather in Phoenix?
Sure, I'll look that up for you. Let me just check the current weather conditions in Phoenix.>>>get_weather
{"location": "Phoenix"}<|eot_id|>
{"temperature": "30C"}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
The current weather in Phoenix is 30C.<|eot_id|>
What is 38484 + 323?
Sure, let's calculate that.>>>calculate
{"expression": "38484 + 323"}<|eot_id|>
{"result": 38807}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
The sum of 38484 and 323 is 38807.<|eot_id|>
What is 67 feet in meters?
To convert 67 feet into meters, we use the conversion factor: 1 foot is approximately 0.3048 meters. Let's calculate it.>>>calculate
{"expression": "67 * 0.3048"}<|eot_id|>
{"result": 20.4216}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
67 feet is approximately 20.4216 meters.<|eot_id|>
```

## Function calling example, hiding details
```
./examples/function-calling/llama-cli-function-runner.py  -m `huggingface-cli  download meetkai/functionary-small-v3.2-GGUF functionary-small-v3.2.Q4_0.gguf` -i
What is the weather in Phoenix?
To provide you with the current weather in Phoenix, Arizona, I will need to check the weather data for you. Let me get that information.
The current weather in Phoenix, Arizona is 30°C. If you have any more questions about weather in other locations, feel free to ask!
Is it colder in Vegas?
To determine if the current temperature in Las Vegas is colder than in Phoenix, which is currently 30°C, I will need to check the weather data for Las Vegas. Let's find out.
The current weather in Las Vegas, Nevada is also 30°C. Therefore, there is no difference in temperature between Phoenix and Las Vegas at the moment. If you have any more questions or need further assistance, please let me know!
What is 37234 times 39?
To calculate 37234 times 39, I'll perform the multiplication. Let's do that.
The result of multiplying 37234 by 39 is 1,452,126. If you have any more calculations or questions, feel free to ask!
```

## Function calling example, using Phi-3 function calling
```
./examples/function-calling/llama-cli-function-runner.py -m `huggingface-cli download nold/Phi-3-mini-4k-instruct-function-calling-GGUF Phi-3-mini-4k-instruct-function-calling_Q4_K_M.gguf` --special --display-prompt -i
```
