/**
 * Detects whether a model's chat template supports thinking/reasoning control.
 *
 * The server "/props" endpoint does NOT expose a supports_thinking flag.
 * It is computed internally by common_chat_templates_support_enable_thinking
 * in common/chat.cpp. A proper server flag would make this unnecessary.
 *
 * Detection order (most reliable first):
 *  1. Thinking-control Jinja2 variables  === pass-through via chat_template_kwargs
 *  2. Thinking-control Jinja2 conditionals  === template-native on/off logic
 *  3. Paired thinking-content tag pairs    === models that output special tags
 */

const THINKING_KWARG_VARS = ['enable_thinking', 'reasoning_effort', 'thinking_budget'];

/**
 * Paired thinking-content tag patterns.
 *
 * Inspected: llama-cpp-deepseek-r1/v3, nim-nemotron-{3,4}-nano, qwen-qwq-32b,
 * qwen-3-32b, google-gemma-4-31b-it, kimikimi-k2-thinking, apertus-8b-instruct,
 * mistralai-Mistral-Small-3.2-24B, ByteDance-Seed-OSS.
 *
 * The self-closing entry is Kimi-K2, Gemma4 fixed-length pair,
 * where both tags always appear adjacent with no content between.
 */
const THINKING_TAG_PATTERNS: Array<[string, string | null]> = [
	['<think>', '</think>'],
	['<|channel>thought', '<|channel|>'],
	['<|think|>', '</|think|>'],
	['<seed:think|>', '</seed:think|>'],
	['<think></think>', null]
];

const JINJA_THINKING_CONDITIONALS: RegExp[] = [
	// Matches: {% if enable thinking %}, {% if enable_thinking %}, {% if (enable_thinking is defined) %}
	// Handles: underscore-separated (enable_thinking), space-separated (enable thinking),
	// and optional parens/brackets before enable (if (enable_thinking )
	/\{%-?\s*if\s+\(?\s*\w*enable[\s_]+\w*(thinking|think|reasoning)/i,
	/\{%-?\s*if\s+\w*(thinking|reasoning)\s*(is not|==|!=)/i,
	/\{%-?\s*if\s+not\s+\w*enable/i,
	/\{%-?\s*if\s+ns\.enable_thinking/i
];

/** Guards against false positives:
 *  - Generic thought keyword (tool descriptions say "chain of thought")
 *  - Qwen vertical-bar token (used for ALL tool calls, not thinking)
 */
export function detectThinkingSupport(t: string): boolean {
	if (!t) return false;
	for (const kwarg of THINKING_KWARG_VARS) {
		const regex = new RegExp(
			`(\\{\\{[^{}]*\\b${kwarg}\\b[^{}]*\\}\\}|\\{%[^{}]*\\b${kwarg}\\b[^{}]*%\\})`,
			'i'
		);
		if (regex.test(t)) return true;
	}
	for (const p of JINJA_THINKING_CONDITIONALS) {
		if (p.test(t)) return true;
	}
	for (const [s, e] of THINKING_TAG_PATTERNS) {
		if (t.includes(s) && (!e || t.includes(e))) return true;
	}
	return false;
}

export function detectThinkingSupportWithReason(t: string): { supported: boolean; reason: string } {
	if (!t) return { supported: false, reason: 'No chat template available' };
	for (const kwarg of THINKING_KWARG_VARS) {
		const regex = new RegExp(
			`(\\{\\{[^{}]*\\b${kwarg}\\b[^{}]*\\}\\}|\\{%[^{}]*\\b${kwarg}\\b[^{}]*%\\})`,
			'i'
		);
		if (regex.test(t)) {
			return { supported: true, reason: 'Found: ' + kwarg };
		}
	}
	for (const p of JINJA_THINKING_CONDITIONALS) {
		if (p.test(t)) return { supported: true, reason: 'Found: thinking conditional' };
	}
	for (const [s, e] of THINKING_TAG_PATTERNS) {
		if (t.includes(s) && (!e || t.includes(e))) {
			return { supported: true, reason: 'Found: ' + s + (e ? ' .. ' + e : ' (self)') };
		}
	}
	return { supported: false, reason: 'No thinking patterns found' };
}
