import json, os

base = os.path.dirname(os.path.abspath(__file__))

lines = open(os.path.join(base, 'rwsft-training-data.jsonl'), encoding='utf-8').readlines()
split = int(len(lines) * 0.95)

train_lines = lines[:split]
val_lines   = lines[split:]

train_out = os.path.join(base, 'ppl-eval-train.txt')
val_out   = os.path.join(base, 'ppl-eval-val.txt')

def fmt(s):
    # Full prompt+response so the model is conditioned correctly.
    # llama-perplexity scores all tokens, but the prompt PPL is identical
    # for base vs adapter — the delta is driven by the response tokens.
    prompt   = s.get('prompt', '').strip()
    response = s.get('response', '').strip()
    if not response:
        return None
    if prompt:
        return prompt + '\n' + response
    return response

with open(train_out, 'w', encoding='utf-8') as f:
    for line in train_lines:
        text = fmt(json.loads(line))
        if text:
            f.write(text + '\n\n')

with open(val_out, 'w', encoding='utf-8') as f:
    for line in val_lines:
        text = fmt(json.loads(line))
        if text:
            f.write(text + '\n\n')

train_chars = len(open(train_out, encoding='utf-8').read())
val_chars   = len(open(val_out,   encoding='utf-8').read())
print(f'train: {len(train_lines)} samples, {train_chars:,} chars -> ppl-eval-train.txt')
print(f'val:   {len(val_lines)} samples,  {val_chars:,} chars  -> ppl-eval-val.txt')
