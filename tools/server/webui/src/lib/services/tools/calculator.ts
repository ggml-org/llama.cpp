import type { ApiToolDefinition } from '$lib/types';
import { registerTool } from './registry';

export const CALCULATOR_TOOL_NAME = 'calculator';

export const calculatorToolDefinition: ApiToolDefinition = {
	type: 'function',
	function: {
		name: CALCULATOR_TOOL_NAME,
		description:
			'Safely evaluate a math expression. Supports operators + - * / % ^ and parentheses. Functions: sin, cos, tan, asin, acos, atan, atan2, sqrt, abs, exp, ln/log/log2, max, min, floor, ceil, round, pow. Constants: pi, e. Angles are in radians.',
		parameters: {
			type: 'object',
			properties: {
				expression: {
					type: 'string',
					description:
						'Math expression using numbers, parentheses, +, -, *, /, %, and exponentiation (^).'
				}
			},
			required: ['expression']
		}
	}
};

// Allow digits, letters (for functions/constants), commas, whitespace, and math operators
const SAFE_EXPRESSION = /^[0-9+*/().,^%\sA-Za-z_-]*$/;
const ALLOWED_IDENTIFIERS = new Set([
	'sin',
	'cos',
	'tan',
	'asin',
	'acos',
	'atan',
	'atan2',
	'sqrt',
	'abs',
	'exp',
	'ln',
	'log',
	'log2',
	'max',
	'min',
	'floor',
	'ceil',
	'round',
	'pow',
	'pi',
	'e'
]);

function rewriteFunctions(expr: string): string {
	// Map identifiers to Math.* (or constants)
	const replacements: Record<string, string> = {
		sin: 'Math.sin',
		cos: 'Math.cos',
		tan: 'Math.tan',
		asin: 'Math.asin',
		acos: 'Math.acos',
		atan: 'Math.atan',
		atan2: 'Math.atan2',
		sqrt: 'Math.sqrt',
		abs: 'Math.abs',
		exp: 'Math.exp',
		ln: 'Math.log',
		log: 'Math.log',
		log2: 'Math.log2',
		max: 'Math.max',
		min: 'Math.min',
		floor: 'Math.floor',
		ceil: 'Math.ceil',
		round: 'Math.round',
		pow: 'Math.pow'
	};

	let rewritten = expr;

	for (const [id, replacement] of Object.entries(replacements)) {
		// only match bare function names not already qualified (no letter/number/_ or dot before)
		const re = new RegExp(`(^|[^A-Za-z0-9_\\.])${id}\\s*\\(`, 'g');
		rewritten = rewritten.replace(re, `$1${replacement}(`);
	}

	rewritten = rewritten.replace(/\bpi\b/gi, 'Math.PI').replace(/\be\b/gi, 'Math.E');

	return rewritten;
}

export function evaluateCalculatorExpression(expr: string): string {
	const trimmed = expr.trim();
	if (trimmed.length === 0) {
		return 'Error: empty expression.';
	}
	if (!SAFE_EXPRESSION.test(trimmed)) {
		return 'Error: invalid characters. Allowed: digits, + - * / % ^ ( ) , and basic function names.';
	}

	// Check identifiers are allowed
	const identifiers = trimmed.match(/[A-Za-z_][A-Za-z0-9_]*/g) || [];
	for (const id of identifiers) {
		if (!ALLOWED_IDENTIFIERS.has(id.toLowerCase())) {
			return `Error: unknown identifier "${id}".`;
		}
	}

	try {
		// Replace caret with JS exponent operator
		const caretExpr = trimmed.replace(/\^/g, '**');
		// Rewrite functions/constants to Math.*
		const rewritten = rewriteFunctions(caretExpr);
		const result = Function(`"use strict"; return (${rewritten});`)();

		if (typeof result === 'number' && Number.isFinite(result)) {
			return result.toString();
		}

		return 'Error: expression did not produce a finite number.';
	} catch (err) {
		return `Error: ${err instanceof Error ? err.message : 'failed to evaluate expression.'}`;
	}
}

registerTool({
	name: CALCULATOR_TOOL_NAME,
	label: 'Calculator',
	description:
		'Safely evaluate a math expression using basic operators and Math functions (client-side).',
	enableConfigKey: 'enableCalculatorTool',
	defaultEnabled: true,
	definition: calculatorToolDefinition,
	execute: async (argsJson: string) => {
		let expression = argsJson;
		try {
			const parsedArgs = JSON.parse(argsJson);
			if (parsedArgs && typeof parsedArgs === 'object' && 'expression' in parsedArgs) {
				expression = parsedArgs.expression as string;
			}
		} catch {
			// ignore
		}
		const result = evaluateCalculatorExpression(expression);
		return { content: result };
	}
});
