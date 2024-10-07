import os

i = 0
prompt_list = []
with open("wikitext-2-raw/wiki.test.raw", "r", encoding='utf-8') as file:
    for line in file:
        # if not line.startswith(' =') and i < 1000:
        if not (line.startswith(' \n') or line.startswith(' =')):
            i = i + 1
            prompt_list.append(line[1:-1])

with open("prompt.txt", "w", encoding='utf-8') as file:
    for prompt in prompt_list:
        file.write(prompt)
        file.write('\n')

# for prompt in prompt_list:
#     system_code = ".\\build\\bin\\Release\\llama-cli.exe -m ..\\..\\gguf\\ggml-tq20-700m.gguf -b 1 -t 4 --seed 0 -p \"{}\"".format(prompt)
    # print(system_code)
    # os.system(system_code)