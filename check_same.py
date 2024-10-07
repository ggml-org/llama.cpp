i2_s = []
with open("i2s.txt", "r", encoding='utf-8', errors='ignore') as file:
    for line in file:
        line = "<#>" + line[4:]
        i2_s.append(line.split("<#>")[1:])

tq20 = []
with open("tq20.txt", "r", encoding='utf-8', errors='ignore') as file:
    for line in file:
        line = "<#>" + line[4:]
        tq20.append(line.split("<#>")[1:])

i = 0
prompt_list = []
with open("prompt.txt", "r", encoding='utf-8', errors='ignore') as file:
    for line in file:
        prompt_list.append(line)

prompt_length = []
for index in range(1000):
    token_num = 0
    temp_p = str(prompt_list[index])
    for token in temp_p:
        if token in ['\n', 'ã', '‑', '©', '₱', 'Ō', 'ñ', 'ǜ', 'ä', 'ö', 'ō', 'í', 'ú', 'ō', 'ū', '⁄', '’', '¥', '°', '−', '–', '—', 'é', '‡', '†', 'ó', 'á', 'ô', 'è', 'ü', 'É', '→', 'α']:
            temp_p = temp_p.replace(token, "", 1)
            token_num = token_num + 1
    for token in tq20[index]:
        if len(temp_p) == 0:
            break
        # if index == 994:
        #     print(temp_p)
        #     print(token)
        temp_p = temp_p.replace(token, "", 1)
        # if index == 994:
        #     print(temp_p)
        #     print(len(temp_p))
        #     print("check")
        token_num = token_num + 1
    # if index == 998:
    #     print(token_num)
    #     print(temp_p)
    if len(temp_p) > 0:
        print("wrong {}".format(index))
        print(temp_p)
        prompt_length.append(-1)
    else:
        prompt_length.append(token_num)
    

# for i in range(len(prompt_length)):
#     print(prompt_length[i])

# i2_s = i2_s[1:]
# tq20 = tq20[1:]

# # print(len(i2_s))
# # print(len(tq20))

min_10 = 0
min_20 = 0
min_30 = 0
min_40 = 0
min_50 = 0
min_60 = 0
min_70 = 0
min_80 = 0
min_90 = 0
min_100 = 0
clear = 0

same_token = []

for index in range(1000):
    ptr = 0
    same_num = 0
    if prompt_length[index] == -1:
        same_token.append(-1)
        continue
    for ptr in range(min(len(i2_s[index]), len(tq20[index]))):
        if i2_s[index][ptr] != tq20[index][ptr]:
            same_token.append(same_num + 2 - prompt_length[index])
            break
        else:
            if len(i2_s) == len(tq20) and ptr == len(i2_s) - 1:
                print("clear")
            same_num = same_num + 1

print(same_token)
    # tq20_i = 0
    # i2_s_i = 0
    # print(len(i2_s[i]))
    # print(len(tq20[i]))
    
    # while (tq20_i < len(tq20[i]) and i2_s_i < len(i2_s[i]) and i2_s[i][i2_s_i] == tq20[i][tq20_i]):
    #     tq20_i = tq20_i + 1
    #     i2_s_i = i2_s_i + 1
    #     if tq20_i == len(tq20[i]) - 1 and i2_s_i == len(i2_s[i]) - 1:
    #         print(i)
    #         clear = clear + 1
    #     else:
    #         # print("token {}, index {}".format(i, j))
    #         # print("i2_s {}, tq20 {}", i2_s[i][j], tq20[i][j])
    #         if j - 4 < 5:
    #         min_10 = min_10 + 1
    #     elif j - 4 < 10:
    #         min_20 = min_20 + 1
    #     # elif j < 30:
    #     #     min_30 = min_30 + 1
    #     # elif j < 40:
    #     #     min_40 = min_40 + 1
    #     elif j - 4 < 50:
    #         min_50 = min_50 + 1
    #     # elif j < 60:
    #     #     min_60 = min_60 + 1
    #     # elif j < 70:
    #     #     min_70 = min_70 + 1
    #     # elif j < 80:
    #     #     min_80 = min_80 + 1
    #     # elif j < 90:
    #     #     min_90 = min_90 + 1
    #     elif j - 4 < 100:
    #         min_100 = min_100 + 1
            

# # print(min_10)
# # print(min_20)
# # print(min_30)
# # print(min_40)
# # print(min_50)
# # print(min_60)
# # print(min_70)
# # print(min_80)
# # print(min_90)
# # print(min_100)
# print(clear)
