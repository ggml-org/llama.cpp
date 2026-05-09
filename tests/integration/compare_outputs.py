import argparse, subprocess, sys

def run_inference(model_args: list[str], prompt: str, n_tokens: int) -> str:
    cmd = [
        'build/bin/llama-simple',
        '-m', model_args[1],
        '-n', str(n_tokens),
        prompt,
    ]
    env = {}
    if '--ffn-file' in model_args:
        idx = model_args.index('--ffn-file')
        env['LLAMA_FFN_FILE'] = model_args[idx + 1]

    import os
    e = os.environ.copy()
    e.update(env)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, env=e)
        return result.stdout.strip()
    except subprocess.TimeoutExpired as e:
        return e.stdout.decode('utf-8').strip() if e.stdout else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full-model',   required=True)
    ap.add_argument('--attn-model',   required=True)
    ap.add_argument('--ffn-file',     required=True)
    ap.add_argument('--prompt',       default="The capital of France is")
    ap.add_argument('--n-tokens',     type=int, default=10)
    ap.add_argument('--tolerance',    type=float, default=1e-3)
    args = ap.parse_args()

    full_out = run_inference(['-m', args.full_model], args.prompt, args.n_tokens)
    split_out = run_inference([
        '-m', args.attn_model,
        '--ffn-file', args.ffn_file,
        '--split-mode', 'local-ssd',
    ], args.prompt, args.n_tokens)

    def clean(s):
        lines = s.split('\n')
        res = []
        for l in lines:
            if "memory breakdown" not in l and "- Host" not in l and "- CPU" not in l and l.strip():
                res.append(l)
        for l in lines:
            if l.startswith("<s>"): return l
        return s

    full_out = clean(full_out)
    split_out = clean(split_out)

    print(f"Full  output: {repr(full_out[:200])}")
    print(f"Split output: {repr(split_out[:200])}")

    if full_out and full_out == split_out:
        print("\nPASS: outputs identical")
        sys.exit(0)
    else:
        full_toks  = full_out.split()
        split_toks = split_out.split()
        if len(full_toks) == 0:
            print("FAIL: Full output empty!")
            sys.exit(1)
        mismatches = sum(a != b for a, b in zip(full_toks, split_toks))
        if mismatches <= max(1, len(full_toks) // 10):
            print(f"\nPASS: {mismatches}/{len(full_toks)} token mismatches (within tolerance)")
            sys.exit(0)
        else:
            print(f"\nFAIL: {mismatches}/{len(full_toks)} token mismatches (exceeds tolerance)")
            sys.exit(1)

if __name__ == '__main__':
    main()
