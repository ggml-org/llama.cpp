export function parseExecShellCommandError(
	toolResultString: string | undefined
): string | undefined {
	if (!toolResultString) return undefined;

	// Exec results are usually large plain-text stdout; only a JSON object
	// root can carry an error field, so skip the parse otherwise
	const trimmed = toolResultString.trimStart();

	if (trimmed[0] !== '{') return undefined;

	try {
		const parsed: unknown = JSON.parse(trimmed);

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
