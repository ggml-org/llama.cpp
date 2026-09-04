#!/bin/bash
# MTP head calibration A/B. Design and results: MTP_imatrix_AB_test.md
#
# One imatrix pass with the head, five quantizations that differ only in block 64,
# a per-tensor hash proof that the trunks are identical, then teacher-forced
# head-vs-trunk agreement on a held-out text for every variant.
#
# Needs llama-imatrix with --process-mtp, llama-quantize and llama-mtp-agree
# from this tree on PATH. Every step is skipped when its output already exists,
# so a stopped run resumes where it left off; delete an output to redo a step.
set -e

SRC=${SRC:-qwen3.8-27b-bf16.gguf}
CAL=${CAL:-calibration_datav3.txt}
IMX=${IMX:-kld-base/qwen3.8-27b-imatrix-cal-mtp.gguf}
OUT=${OUT:-qwen3.8-27b-q6_k-gateup-q5_k}
EVAL_SRC=${EVAL_SRC:-val.jsonl.zst}
EVAL=${EVAL:-tmp/mtp-agree-val.txt}
EVAL_BYTES=${EVAL_BYTES:-600000}
PCORES=${PCORES:-0,2,4,6,8,10,12,14}
NGL_IMATRIX=${NGL_IMATRIX:-20}
VARIANTS=${VARIANTS:-A B C D E}

# variant letter -> block 64 type; A is q4_0 with the head's imatrix entries excluded
head_type() {
    case $1 in
        A|B) echo q4_0 ;;
        C)   echo bf16 ;;
        D)   echo iq3_s ;;
        E)   echo iq2_xs ;;
        *)   echo "unknown variant $1" >&2; exit 1 ;;
    esac
}

step() { echo; echo "== $*"; }

mkdir -p tmp kld-base

step "imatrix with the head: $IMX"
if [ -s "$IMX" ]; then
    echo "exists, skipping"
else
    taskset -c "$PCORES" \
        llama-imatrix -m "$SRC" -f "$CAL" -o "$IMX" \
        -dev CUDA0 -ngl "$NGL_IMATRIX" \
        -c 5120 -np 1 -b 5120 -ub 5120 \
        --no-ppl --process-output --process-mtp
fi
python gguf-py/gguf/scripts/gguf_dump.py --no-tensors "$IMX" | grep -E 'imatrix\.(datasets|chunk_count|chunk_size)'
echo "head entries: $(python gguf-py/gguf/scripts/gguf_dump.py "$IMX" | grep -c 'blk\.64\.') (expect 16)"

step "held-out text: $EVAL"
if [ -s "$EVAL" ]; then
    echo "exists, skipping"
else
    zstdcat "$EVAL_SRC" | jq -r .text | head -c "$EVAL_BYTES" > "$EVAL"
fi

for v in $VARIANTS; do
    types=tmp/quant-types-mtp-$v.txt
    printf '%s\n' "blk\\.64\\.=$(head_type "$v")" '\.ffn_gate\.weight$=q5_k' '\.ffn_up\.weight$=q5_k' > "$types"

    exclude=()
    [ "$v" = A ] && exclude=(--exclude-weights blk.64.)

    step "quantize $v: block 64 $(head_type "$v")${exclude:+, no head imatrix}"
    if [ -s "$OUT-mtp$v.gguf" ]; then
        echo "exists, skipping"
    else
        llama-quantize --imatrix "$IMX" --pure --tensor-type-file "$types" "${exclude[@]}" "$SRC" "$OUT-mtp$v.gguf" Q6_K > "tmp/quantize-mtp$v.log"
        grep 'blk\.64\.' "tmp/quantize-mtp$v.log" | head -3
    fi
done

step "trunk identity: every tensor outside block 64 must hash equal to A"
for v in $VARIANTS; do
    [ -s "tmp/hash-mtp$v.txt" ] || python gguf-py/gguf/scripts/gguf_hash.py "$OUT-mtp$v.gguf" | sed -E 's/  [^ ]+:/  /' > "tmp/hash-mtp$v.txt"
done
trunk() { grep -v 'blk\.64\.' "tmp/hash-mtp$1.txt" | grep -v 'gguf$'; }
for v in $VARIANTS; do
    n=$(diff <(trunk A) <(trunk "$v") | grep -c '^>' || true)
    echo "$v: $n trunk tensors differ from A"
    [ "$n" = 0 ] || { echo "trunk mismatch, stopping" >&2; exit 1; }
done

for v in $VARIANTS; do
    step "agreement $v"
    if [ -s "tmp/mtp-agree-$v.txt" ]; then
        echo "exists, skipping"
    else
        llama-mtp-agree -m "$OUT-mtp$v.gguf" -f "$EVAL" -c 1024 -b 1024 -ngl 99 -fa on > "tmp/mtp-agree-$v.log" 2>&1
        grep -E 'positions|head|proposed|trunk|mean KL' "tmp/mtp-agree-$v.log" | grep -v load_tensors > "tmp/mtp-agree-$v.txt"
        # per-chunk "agree N of M" lines -> one "N M" row per chunk
        sed -nE 's/.*chunk [0-9]+\/[0-9]+: agree ([0-9]+) of ([0-9]+).*/\1 \2/p' "tmp/mtp-agree-$v.log" > "tmp/mtp-agree-$v.chunks"
    fi
    cat "tmp/mtp-agree-$v.txt"
done

step "paired test against A: per-chunk agreement differences"
printf 'variant\tchunks\tmean_diff\tse\tt\n'
for v in $VARIANTS; do
    [ "$v" = A ] && continue
    paste tmp/mtp-agree-A.chunks "tmp/mtp-agree-$v.chunks" | awk -v v="$v" '
        { d = $3 / $4 - $1 / $2; n++; s += d; ss += d * d }
        END { m = s / n; se = sqrt((ss / n - m * m) / (n - 1)); printf "%s\t%d\t%+.5f\t%.5f\t%+.2f\n", v, n, m, se, m / se }'
done

step "summary"
printf 'variant\thead\tagreement\tproposed\tprecision\thead_hits\tkl\n'
for v in $VARIANTS; do
    awk -v v="$v" -v t="$(head_type "$v")" '
        /head == trunk/    { agree = $NF }
        /proposed at/      { prop = $(NF-6); prec = $(NF-3) }
        /head  hits/       { hits = $NF }
        /mean KL/          { kl = $NF }
        END { printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", v, t, agree, prop, prec, hits, kl }' "tmp/mtp-agree-$v.txt"
done
