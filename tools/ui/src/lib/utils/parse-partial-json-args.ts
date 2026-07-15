// Parse partial tool-arg JSON streamed token-by-token. Closes any
// unterminated string and dangling open containers (in reverse order),
// so parsers can still surface keys already received while the call
// is still in flight.
export function parsePartialJsonArgs(toolArgsString: string): Record<string, unknown> | null {
	try {
		const parsed: unknown = JSON.parse(toolArgsString);
		if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
			return parsed as Record<string, unknown>;
		}
		return null;
	} catch {
		let inString = false;
		let escape = false;
		const stack: ('{' | '[')[] = [];

		for (let i = 0; i < toolArgsString.length; i++) {
			const ch = toolArgsString[i];
			if (escape) {
				escape = false;
				continue;
			}
			if (ch === '\\' && inString) {
				escape = true;
				continue;
			}
			if (ch === '"') {
				inString = !inString;
				continue;
			}
			if (inString) continue;
			if (ch === '{') stack.push('{');
			else if (ch === '}') {
				if (stack.length === 0 || stack[stack.length - 1] !== '{') return null;
				stack.pop();
			} else if (ch === '[') stack.push('[');
			else if (ch === ']') {
				if (stack.length === 0 || stack[stack.length - 1] !== '[') return null;
				stack.pop();
			}
		}

		let completed = toolArgsString;
		if (escape) {
			// Dangling escape at end of partial JSON: escape the trailing
			// backslash as a literal so we can close the string cleanly.
			completed += '\\';
		}
		if (inString) completed += '"';
		if (!inString) completed = completed.replace(/,?\s*$/, '');

		// Close in reverse nesting order: innermost container first.
		for (let i = stack.length - 1; i >= 0; i--) {
			completed += stack[i] === '{' ? '}' : ']';
		}

		try {
			const parsed: unknown = JSON.parse(completed);
			if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
				return parsed as Record<string, unknown>;
			}
			return null;
		} catch {
			return null;
		}
	}
}
