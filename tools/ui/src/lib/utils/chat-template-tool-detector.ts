/**
 * Detects whether a model's chat template supports tool calling.
 *
 * There is no server flag for tool support, so we infer it from the chat
 * template. A template that accepts a `tools` array or emits tool-call tokens
 * is treated as tool-capable.
 */

/** Tool-call tokens emitted by the template for assistant tool calls. */
const TOOL_CALL_TOKENS = [
	'tool_call',
	'tool_calls',
	'function_call',
	'tool_use',
	'<tool',
	'<|tool',
	'TOOL_CALL'
];
/** Jinja reference to the `tools` array passed in by the caller. */
const JINJA_TOOLS_VAR = /\{\{[^{}]*\btools\b[^{}]*\}\}|\{%[^{}]*\btools\b[^{}]*%\}/i;

export function detectToolUseSupport(t: string): boolean {
	if (!t) return false;

	if (JINJA_TOOLS_VAR.test(t)) return true;

	return TOOL_CALL_TOKENS.some((token) => t.includes(token));
}
