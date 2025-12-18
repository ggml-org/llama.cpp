import time
import argparse

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("dir_tokenizer", help="directory containing 'tokenizer.model' file")
parser.add_argument("--fname-tok",   help="path to a text file to tokenize", required=True)
args = parser.parse_args()

dir_tokenizer = args.dir_tokenizer
fname_tok = args.fname_tok

tokenizer = AutoTokenizer.from_pretrained(dir_tokenizer)

print('tokenizing file: ', fname_tok) # noqa: NP100
fname_out = fname_tok + '.tok'
with open(fname_tok, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    s = ''.join(lines)
    t_start = time.time()
    res = tokenizer.encode(s, add_special_tokens=False)
    t_end = time.time()
    print('\nmain : tokenized in', "{:.3f}".format(1000.0 * (t_end - t_start)), 'ms (py)') # noqa: NP100
    with open(fname_out, 'w', encoding='utf-8') as f:
        for x in res:
            # LLaMA v3 for some reason strips the space for these tokens (and others)
            # if x == 662:
            #     f.write(str(x) + ' \' ' + tokenizer.decode(x) + '\'\n')
            # elif x == 1174:
            #     f.write(str(x) + ' \' ' + tokenizer.decode(x) + '\'\n')
            # elif x == 2564:
            #     f.write(str(x) + ' \' ' + tokenizer.decode(x) + '\'\n')
            # elif x == 758:
            #     f.write(str(x) + ' \' ' + tokenizer.decode(x) + '\'\n')
            # elif x == 949:
            #     f.write(str(x) + ' \' ' + tokenizer.decode(x) + '\'\n')
            # elif x == 5354:
            #     f.write(str(x) + ' \' ' + tokenizer.decode(x) + '\'\n')
            # else:
            #     f.write(str(x) + ' \'' + tokenizer.decode(x) + '\'\n')
            # f.write(str(x) + ' \'' + tokenizer.decode(x).strip() + '\'\n')
            f.write(str(x) + '\n')
    print('len(res): ', len(res)) # noqa: NP100
    print('len(lines): ', len(lines)) # noqa: NP100
print('results written to: ', fname_out) # noqa: NP100
