import argparse, subprocess, sys

def run_inference(model_args, prompt, n_tokens):
    cmd = [
        './build/bin/llama-cli',
        '-p', prompt,
        '-n', str(n_tokens),
        '--log-disable',
        *model_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()

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

    print(f"Full  output: {repr(full_out[:200])}")
    print(f"Split output: {repr(split_out[:200])}")

    full_toks  = full_out.split()
    split_toks = split_out.split()
    mismatches = sum(a != b for a, b in zip(full_toks, split_toks))
    if mismatches <= max(1, len(full_toks) // 10):
        print(f"\nPASS: {mismatches}/{len(full_toks)} token mismatches")
        sys.exit(0)
    else:
        print(f"\nFAIL: {mismatches}/{len(full_toks)} token mismatches")
        sys.exit(1)

if __name__ == '__main__':
    main()
