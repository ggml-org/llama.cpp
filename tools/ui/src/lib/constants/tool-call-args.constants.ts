// Tool-args and tool-result parsing helpers: the file tools' path field
// aliases, the JSON container gates for result blobs, and the targeted
// string-field pattern used for cheap title-tier extraction.

/**
 * Field aliases the file tools accept for the path argument. Tool contracts
 * drifted over time: some models emit `file_path` / `filePath`.
 */
export const TOOL_ARG_PATH_KEYS: readonly string[] = ['path', 'file_path', 'filePath'];

/** Opening character of a JSON object; only an object root can carry fields. */
export const JSON_OBJECT_OPEN = '{';

/** Opening character of a JSON array; successful sandbox output is one. */
export const JSON_ARRAY_OPEN = '[';

/**
 * Matches `"<key>": "<value>"` in a JSON args blob ( whitespace between
 * tokens allowed ), capturing the raw string literal so only that literal
 * gets decoded; escaped quotes stay inside the value group. `{key}` is
 * replaced with the field name before use.
 */
export const TOOL_ARG_STRING_FIELD_PATTERN_TEMPLATE = '"{key}"\\s*:\\s*"((?:[^"\\\\]|\\\\.)*)"';
