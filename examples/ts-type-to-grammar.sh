#!/usr/bin/env bash
#
# ./examples/ts-type-to-grammar.sh "{a:string,b:string,c?:string}"
# python examples/json_schema_to_grammar.py https://json.schemastore.org/tsconfig.json
#
set -euo pipefail

readonly type="$1"

# Create a temporary directory
TMPDIR=""
trap 'rm -fR "$TMPDIR"' EXIT
TMPDIR=$(mktemp -d)

DTS_FILE="$TMPDIR/type.d.ts"
SCHEMA_FILE="$TMPDIR/schema.json"

echo "export type MyType = $type" > "$DTS_FILE"

# This is a fork of typescript-json-schema, actively maintained as of March 2024:
# https://github.com/vega/ts-json-schema-generator
npx ts-json-schema-generator --unstable --no-top-ref --path "$DTS_FILE" --type MyType -e none > "$SCHEMA_FILE"

# Alternative, not actively maintained as of March 2024:
# https://github.com/YousefED/typescript-json-schema
# npx typescript-json-schema --defaultProps --required "$DTS_FILE" MyType | tee "$SCHEMA_FILE" >&2

./examples/json_schema_to_grammar.py "$SCHEMA_FILE"
