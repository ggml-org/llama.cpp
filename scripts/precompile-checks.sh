#!/bin/bash

# This runs some pre compilation sanity checks that certain project rules and guidelines are kept
# This will not contain any signifiant logic, but mostly just obvious and easily greppable checks

ERROR_FOUND=0


## START OF INCLUDES EXCLUDED FROM EXAMPLES FOLDER ##
SRC_DIR="./examples"
FORBIDDEN_HEADERS=("llama-impl.h")
echo "🔍 Scanning for forbidden includes in $SRC_DIR..."
for HEADER in "${FORBIDDEN_HEADERS[@]}"; do
    MATCHES=$(grep -rn --include=\*.{c,cpp} "#include \"$HEADER\"" "$SRC_DIR" 2>/dev/null)

    if [[ -n "$MATCHES" ]]; then
        echo "❌ Forbidden include detected: $HEADER"
        echo "$MATCHES" | while IFS=: read -r FILE LINE _; do
            echo "::error file=$FILE,line=$LINE::Forbidden include: $HEADER in $FILE at line $LINE"
        done
        ERROR_FOUND=1
    fi

done
## END OF INCLUDES EXCLUDED FROM EXAMPLES FOLDER ##


if [[ "$ERROR_FOUND" -eq 1 ]]; then
    echo "❌ Forbidden includes found. Please remove!"
    exit 1
else
    echo "✅ No forbidden includes found."
fi
