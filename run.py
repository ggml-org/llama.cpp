import os

# Llama ASCII art
print("""
                        ▓▓  ▓▓
                      ▓▓░░▓▓░░▓▓
                    ▓▓▓▓░░░░░░▓▓
                  ▓▓░░░░░░██░░▓▓
                  ▓▓░░░░░░░░░░▓▓
                    ▓▓▓▓░░░░░░▓▓
                        ▓▓░░░░▓▓
                        ▓▓░░░░▓▓                ▓▓
                        ▓▓░░░░▓▓              ▓▓░░▓▓
                        ▓▓░░░░▓▓              ▓▓░░▓▓
                        ▓▓░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░▓▓
                        ▓▓░░░░░░░░░░░░░░░░░░░░░░▓▓
                        ▓▓░░░░░░░░░░░░░░░░░░░░░░▓▓
                        ▒▒░░░░░░░░░░░░░░░░░░░░░░▒▒
                        ▓▓░░░░░░░░░░░░░░░░░░░░░░▓▓
                          ▓▓░░░░░░░░░░░░░░░░░░▓▓
                          ▓▓▒▒▒▒░░▓▓▓▓▒▒▒▒▒▒░░▓▓
                          ▓▓░░▓▓░░▓▓  ▓▓░░▓▓░░▓▓
                          ▓▓░░▓▓░░▓▓  ▓▓░░▓▓░░▓▓
                          ▓▓░░▓▓░░▓▓  ▓▓░░▓▓░░▓▓
                          ▓▓░░▓▓░░▓▓  ▓▓░░▓▓░░▓▓
                          ▓▓░░▒▒░░▒▒  ▒▒░░▓▓░░▓▓
                          ░░▒▒  ▒▒░░    ▒▒░░▒▒

""")
print("Welcome to the Llama.cpp\n")


# Set the directory to search in
dir_path = "./models/"

# Initialize a list to store the paths of all .bin files found
bin_files = []

# Traverse the directory and its subdirectories to find all .bin files
for root, dirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith(".bin"):
            bin_files.append(os.path.join(root, file))

# Print the list of .bin files found
print("Choose a file:")
for i, file in enumerate(bin_files):
    print(f"{i+1}. {file}")

# Ask the user to choose a file by entering its number
while True:
    try:
        choice = int(input("Enter the number of the file you want to choose: "))
        if choice < 1 or choice > len(bin_files):
            raise ValueError
        break
    except ValueError:
        print("Invalid choice. Please enter a number between 1 and", len(bin_files))

# Get the path of the chosen file
chosen_file_path = bin_files[choice-1]

# Ask the user for the CTX size
ctx_size = input("Enter the CTX size (default is 2048): ")

# Set the default CTX size to 2048 if no answer is provided
if not ctx_size:
    ctx_size = "2048"

# Ask the user for the Top K
top_k = input("Enter the Top K (default is 10000): ")

# Set the default Top K to 10000 if no answer is provided
if not top_k:
    top_k = "10000"

# Ask the user for the Repeat Penalty
repeat_penalty = input("Enter the Repeat Penalty (default is 1): ")

# Set the default Repeat Penalty to 1 if no answer is provided
if not repeat_penalty:
    repeat_penalty = "1"

# Do something with the chosen file, CTX size, Top K, and Repeat Penalty...


# Ask the user for the temperature
temperature = input("Enter the temperature (between 0 and 2, default is 0.2): ")
if not temperature:
    temperature = "0.2"
else:
    try:
        temperature = float(temperature)
    except ValueError:
        print("Invalid temperature. Using default value of 0.2")
        temperature = "0.2"
    else:
        if temperature < 0 or temperature > 2:
            print("Invalid temperature. Using default value of 0.2")
            temperature = "0.2"




os.system(f"./main -m {chosen_file_path} --color -f ./prompts/alpaca.txt --ctx_size {ctx_size} -n -1 -ins -b 256 --top_k {top_k} --temp {temperature} --repeat_penalty {repeat_penalty} -t 7")
