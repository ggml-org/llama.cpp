#!/usr/bin/env bash
# Run one streaming set through several settings and score each one for speed,
# draft acceptance, output stability and translation quality.
#
# This is the engine. For a ready made experiment call a preset instead, such
# as run-towerplus.sh, which fills these in and calls this.
#
#   MODEL=model.gguf SRC=source.txt REF=reference.txt \
#   bash sweep.sh
#
# SRC holds one complete source sentence per line, REF the matching reference
# translations. The stream is derived from SRC by the example itself.
set -u

SELF_DIR=$(cd "$(dirname "$0")" && pwd)

BIN="${BIN:-$SELF_DIR/../../../build/bin/llama-self-spec-bias}"
MODEL="${MODEL:-}"
SRC="${SRC:-}"
REF="${REF:-}"

OUT="${OUT:-$SELF_DIR/out}"

# limit the run to the first N sentences, empty means all of them
N_SENT="${N_SENT:-}"

STREAM_INTERVAL="${STREAM_INTERVAL:-3}"
N_PREDICT="${N_PREDICT:-200}"
NGL="${NGL:-99}"

# sacrebleu tokenizer for the target language, zh and ja must be set
BLEU_TOKENIZE="${BLEU_TOKENIZE:-13a}"

# A setting may carry a hold back suffix, and the two are alternatives:
#
#   <setting>-omaskN   withhold the last N tokens of every partial answer, so
#                      they cannot be drafted from either. This is mask-k in
#                      the usual sense, and it changes decoding.
#   <setting>-dmaskN   leave decoding alone, only hide the last N tokens when
#                      erasure is measured. The display only variant.
#
# e.g. CONFS="baseline bias-02 bias-02-omask3 bias-02-dmask3"
parse_conf() {
    CONF_BASE=$1
    CONF_OMASK=0
    CONF_DMASK=0
    case $1 in
        *-omask*) CONF_BASE=${1%-omask*}; CONF_OMASK=${1##*-omask} ;;
        *-dmask*) CONF_BASE=${1%-dmask*}; CONF_DMASK=${1##*-dmask} ;;
    esac
}

# prompt around the partial source, model specific
PRE="${PRE:-$'<start_of_turn>user\nPlease translate the following English source text to Chinese:\nEnglish: '}"
SUF="${SUF:-$'<end_of_turn>\n<start_of_turn>model\nChinese: '}"

# how the source is cut into requests, see segment.py for the policies
POLICY="${POLICY:-interval}"

# set SCORE=0 to measure speed only, then only the stdlib is needed
SCORE="${SCORE:-1}"

PY="${PY:-python3}"

usage() {
    echo "set MODEL, SRC and REF, for example:" >&2
    echo "  MODEL=model.gguf SRC=source.txt REF=reference.txt bash sweep.sh" >&2
    exit 1
}

[ -n "$MODEL" ] && [ -n "$SRC" ] && [ -n "$REF" ] || usage

if [ ! -x "$BIN" ]; then
    echo "llama-self-spec-bias not found at $BIN, build it or set BIN" >&2
    exit 1
fi

mkdir -p "$OUT"

IN=$OUT/src.txt
if [ -n "$N_SENT" ]; then
    head -n "$N_SENT" "$SRC" > "$IN"
else
    cp "$SRC" "$IN"
fi
echo "source sentences: $(wc -l < "$IN")"

# segmentation is data, so it is decided here and handed to the decoder.
# swap POLICY, or point STREAM at a file some other tool produced.
STREAM="${STREAM:-$OUT/stream.jsonl}"

if [ ! -f "$STREAM" ] || [ "$STREAM" = "$OUT/stream.jsonl" ]; then
    "$PY" "$SELF_DIR/segment.py" \
        --input "$IN" --output "$STREAM" \
        --policy "$POLICY" --n "$STREAM_INTERVAL" --id-prefix "$(basename "$IN")"
fi

# references keyed by id, so scoring never joins by line order
"$PY" - "$IN" "$REF" "$OUT/refs.jsonl" <<'EOF'
import json, sys
from pathlib import Path
src, ref, out = sys.argv[1:4]
stem = Path(src).name
n    = len(Path(src).read_text(encoding="utf-8").splitlines())
refs = Path(ref).read_text(encoding="utf-8").splitlines()
if len(refs) < n:
    sys.exit("reference has %d lines, need %d" % (len(refs), n))
with open(out, "w", encoding="utf-8") as f:
    for i, r in enumerate(refs[:n]):
        f.write(json.dumps({"id": "%s:%d" % (stem, i), "ref": r}, ensure_ascii=False) + "\n")
print("refs: %d" % n)
EOF

# name -> flags
#
# baseline is the reference. no-cache is there to show what caching
# alone buys. greedy-verify follows greedy decoding, the bias settings do not.
flags_for() {
    case $1 in
        no-cache)      echo "--no-draft-reuse --no-prompt-cache-prefix" ;;
        baseline)      echo "--no-draft-reuse" ;;
        greedy-verify) echo "--draft-bias-beta 0.0" ;;
        bias-01)       echo "--draft-bias-beta 0.1" ;;
        bias-02)       echo "--draft-bias-beta 0.2" ;;
        bias-03)       echo "--draft-bias-beta 0.3" ;;
        bias-04)       echo "--draft-bias-beta 0.4" ;;
        *)             echo "" ;;
    esac
}


CONFS="${CONFS:-no-cache baseline greedy-verify bias-02 bias-03}"

# validate up front: flags_for runs in a subshell, so failing inside it would
# only kill the subshell and the run would quietly use default settings
for conf in $CONFS; do
    parse_conf "$conf"
    if [ -z "$(flags_for "$CONF_BASE")" ]; then
        echo "unknown setting '$conf'" >&2
        echo "known: no-cache baseline greedy-verify bias-01 bias-02 bias-03 bias-04" >&2
        exit 1
    fi
done

for name in $CONFS; do
    parse_conf "$name"
    echo "===== $name"
    # shellcheck disable=SC2046
    "$BIN" -m "$MODEL" -ngl "$NGL" \
        -f "$STREAM" -n "$N_PREDICT" --output-mask-k "$CONF_OMASK" \
        --in-prefix "$PRE" --in-suffix "$SUF" \
        -o "$OUT/$name.jsonl" \
        $(flags_for "$CONF_BASE") > "$OUT/$name.log" 2>&1

    grep -E "main: (steps|prompt tokens|  reuse rate|output tokens|  from draft|draft accepted|prompt eval|draft verify|decode|output speed)" "$OUT/$name.log"

    if [ "$SCORE" != "1" ]; then
        continue
    fi

    if ! "$PY" "$SELF_DIR/score.py" \
        --hyp "$OUT/$name.jsonl" --refs "$OUT/refs.jsonl" \
        --bleu-tokenize "$BLEU_TOKENIZE" --display-mask-k "$CONF_DMASK" \
        --out "$OUT/$name.results.json" \
        > "$OUT/$name.score.log" 2>&1; then
        echo "scoring failed, see $OUT/$name.score.log" >&2
        tail -3 "$OUT/$name.score.log" >&2
    fi
    grep -E "records:|Normalized Erasure:|erasure by|BLEU:|COMET:" "$OUT/$name.score.log" | tee -a "$OUT/$name.log"
done

echo
echo "results in $OUT"
