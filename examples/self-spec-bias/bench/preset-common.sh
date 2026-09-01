# Shared body of the run-<model>.sh presets. Sourced, not executed.
#
# A preset declares what is specific to its model and then sources this:
#
#   PRESET_NAME=towerplus
#   PRESET_MODEL=model.gguf    # file name, looked up in models/
#   PRESET_PRE=$'...'          # prompt before the partial source
#   PRESET_SUF=$'...'          # prompt after it, {src} and {tgt} become the
#                              # language names
#   PRESET_MODEL_HELP="where to get the weights"
#   source "$(dirname "$0")/preset-common.sh"
#
# Everything below is common to every preset: FLORES-200, cut every 3 words,
# scored the same way. TGT_LANG picks the language pair and with it the test
# files, the language names in the prompt and the BLEU tokenizer. Anything else
# can be overridden from the environment.
#
#   TGT_LANG=de bash run-towerplus.sh
set -u

# key -> flores200 file code, name used in the prompt, sacrebleu tokenizer.
# Erasure does not appear here, it uses one multilingual tokenizer for every
# language so that the numbers stay comparable across pairs.
lang_info() {
    case "$1" in
        en) echo "eng_Latn|English|13a"       ;;
        zh) echo "zho_Hans|Chinese|zh"        ;;
        de) echo "deu_Latn|German|13a"        ;;
        ja) echo "jpn_Jpan|Japanese|ja-mecab" ;;
        ko) echo "kor_Hang|Korean|ko-mecab"   ;;
        fr) echo "fra_Latn|French|13a"        ;;
        es) echo "spa_Latn|Spanish|13a"       ;;
        it) echo "ita_Latn|Italian|13a"       ;;
        pt) echo "por_Latn|Portuguese|13a"    ;;
        ru) echo "rus_Cyrl|Russian|13a"       ;;
        *)  echo ""                           ;;
    esac
}

SRC_LANG="${SRC_LANG:-en}"
TGT_LANG="${TGT_LANG:-zh}"

for lang in "$SRC_LANG" "$TGT_LANG"; do
    if [ -z "$(lang_info "$lang")" ]; then
        echo "unknown language '$lang', add it to lang_info in preset-common.sh" >&2
        exit 1
    fi
done

IFS='|' read -r SRC_CODE SRC_NAME SRC_BLEU <<EOF
$(lang_info "$SRC_LANG")
EOF
IFS='|' read -r TGT_CODE TGT_NAME TGT_BLEU <<EOF
$(lang_info "$TGT_LANG")
EOF

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# models are looked up by file name in models/, which is not in git. Put the
# gguf there or symlink it, or set MODEL to a path of your own.
MODELS_DIR="${MODELS_DIR:-$SELF_DIR/models}"

MODEL="${MODEL:-$MODELS_DIR/$PRESET_MODEL}"

# FLORES-200 devtest is only the default. Point SRC and REF anywhere, one
# sentence per line, same line order. Nothing is fetched when you do.
FLORES_DIR="${FLORES_DIR:-$SELF_DIR/data/flores200}"

SRC_GIVEN="${SRC+yes}"
REF_GIVEN="${REF+yes}"

SRC="${SRC:-$FLORES_DIR/$SRC_CODE.devtest}"
REF="${REF:-$FLORES_DIR/$TGT_CODE.devtest}"

# NO_FETCH=1 to never touch the network
NO_FETCH="${NO_FETCH:-0}"

# {src} and {tgt} let one preset serve every language pair
PRE="${PRE:-$PRESET_PRE}"
SUF="${SUF:-$PRESET_SUF}"

PRE="${PRE//\{src\}/$SRC_NAME}"; PRE="${PRE//\{tgt\}/$TGT_NAME}"
SUF="${SUF//\{src\}/$SRC_NAME}"; SUF="${SUF//\{tgt\}/$TGT_NAME}"

export PRE SUF

export STREAM_INTERVAL="${STREAM_INTERVAL:-3}"
export POLICY="${POLICY:-interval}"
export N_PREDICT="${N_PREDICT:-200}"
export BLEU_TOKENIZE="${BLEU_TOKENIZE:-$TGT_BLEU}"
export CONFS="${CONFS:-baseline greedy-verify bias-02 bias-03}"

export MODEL SRC REF
export OUT="${OUT:-$SELF_DIR/out/$PRESET_NAME-$SRC_LANG$TGT_LANG}"

if [ "${QUICK:-0}" = "1" ]; then
    export N_SENT="${N_SENT:-20}"
else
    export N_SENT="${N_SENT:-}"
fi

# only the default test set is fetched, a path you chose is your own to provide
if [ -n "$SRC_GIVEN" ] || [ -n "$REF_GIVEN" ]; then
    for f in "$SRC" "$REF"; do
        if [ ! -f "$f" ]; then
            echo "missing: $f" >&2
            exit 1
        fi
    done
elif [ ! -f "$SRC" ] || [ ! -f "$REF" ]; then
    if [ "$NO_FETCH" = "1" ]; then
        echo "missing $SRC or $REF, and NO_FETCH=1" >&2
        exit 1
    fi
    FLORES_DIR="$FLORES_DIR" bash "$SELF_DIR/get-flores.sh" "$SRC_CODE" "$TGT_CODE" || exit 1
    echo
fi

if [ ! -f "$MODEL" ]; then
    cat >&2 <<EOF
missing: $MODEL

Put the gguf at that path, symlink it there, or set MODEL to a path of your
own. ${PRESET_MODEL_HELP:-}
EOF
    exit 1
fi

echo "preset : $PRESET_NAME"
echo "langs  : $SRC_NAME to $TGT_NAME"
echo "model  : $MODEL"
echo "source : $SRC"
echo "output : $OUT"
echo

exec bash "$SELF_DIR/sweep.sh"
