import os

with open("prompt.txt", "r", encoding='utf-8') as file:
    for prompt in file:
        system_code = ".\\build\\bin\\Release\\llama-cli.exe -m ..\\..\\gguf\\ggml-i2_s-700m.gguf -b 1 -t 4 --seed 0 -p \"{}\"".format(prompt)
        os.system(system_code)

# for prompt in prompt_list:
#     system_code = ".\\build\\bin\\Release\\llama-cli.exe -m ..\\..\\gguf\\ggml-tq20-700m.gguf -b 1 -t 4 --seed 0 -p \"{}\"".format(prompt)
    # print(system_code)
    # os.system(system_code)