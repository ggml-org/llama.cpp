export function maybeDecodeEscapedMultilineSource(code: string): string | undefined {
	const source = code ?? '';

	if (!source || /[\r\n]/.test(source) || !/\\(?:r|n)/.test(source)) {
		return undefined;
	}

	const decoded = source
		.replace(/\\r\\n/g, '\n')
		.replace(/\\n/g, '\n')
		.replace(/\\r/g, '\n')
		.replace(/\\t/g, '\t');

	return decoded !== source ? decoded : undefined;
}
