export function parseExecShellCommandError(
	toolResultString: string | undefined
): string | undefined {
	if (!toolResultString) return undefined;
	try {
		const parsed: unknown = JSON.parse(toolResultString);
		if (
			parsed &&
			typeof parsed === 'object' &&
			!Array.isArray(parsed) &&
			typeof (parsed as Record<string, unknown>).error === 'string'
		) {
			return (parsed as { error: string }).error;
		}
	} catch {
		// Plain-text result = stdout/stderr, no structured error to surface.
	}
	return undefined;
}
