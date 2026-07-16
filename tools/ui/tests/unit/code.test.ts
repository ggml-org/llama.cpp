import { describe, expect, it } from 'vitest';
import { highlightCode, trimCodePadding } from '$lib/utils/code';

describe('trimCodePadding', () => {
	it('removes a single leading newline', () => {
		expect(trimCodePadding('\nfunction foo() {}')).toBe('function foo() {}');
	});

	it('removes multiple leading newlines', () => {
		expect(trimCodePadding('\n\n\nfunction foo() {}')).toBe('function foo() {}');
	});

	it('removes whitespace-only leading lines', () => {
		expect(trimCodePadding('\n  \n\t\nfunction foo() {}')).toBe('function foo() {}');
	});

	it('removes a single trailing newline', () => {
		expect(trimCodePadding('function foo() {}\n')).toBe('function foo() {}');
	});

	it('removes multiple trailing newlines', () => {
		expect(trimCodePadding('function foo() {}\n\n\n')).toBe('function foo() {}');
	});

	it('removes whitespace-only trailing lines', () => {
		expect(trimCodePadding('function foo() {}\n  \n\t\n')).toBe('function foo() {}');
	});

	it('removes newlines on both sides at once', () => {
		expect(trimCodePadding('\nfunction foo() {}\n')).toBe('function foo() {}');
	});

	it('preserves internal blank lines', () => {
		expect(trimCodePadding('\nfunction foo() {\n\n  return 1;\n}\n')).toBe(
			'function foo() {\n\n  return 1;\n}'
		);
	});

	it('drops a leading whitespace-only line but keeps following code intact', () => {
		expect(trimCodePadding('  \nfunction foo() {}')).toBe('function foo() {}');
	});

	it('passes through already-trimmed input unchanged', () => {
		expect(trimCodePadding('function foo() {}')).toBe('function foo() {}');
		expect(trimCodePadding('function foo() {\n  return 1;\n}')).toBe(
			'function foo() {\n  return 1;\n}'
		);
	});

	it('returns empty string when input is whitespace only', () => {
		expect(trimCodePadding('\n\n\n')).toBe('');
		expect(trimCodePadding('\n  \n\t\n')).toBe('');
	});
});

describe('highlightCode', () => {
	it('returns empty string for empty input', () => {
		expect(highlightCode('', 'javascript')).toBe('');
	});

	it('does not produce a leading newline in the highlighted html', () => {
		const html = highlightCode('\nfunction multiply(a, b) {\n  return a * b;\n}\n', 'javascript');
		expect(html.startsWith('\n')).toBe(false);
		expect(html.startsWith(' ')).toBe(false);
	});

	it('does not produce a trailing newline in the highlighted html', () => {
		const html = highlightCode('\nfunction foo() {}\n', 'javascript');
		expect(html.endsWith('\n')).toBe(false);
	});

	it('preserves internal blank lines in highlighted code', () => {
		const html = highlightCode('\nfunction foo() {\n\n  return 1;\n}\n', 'javascript');
		expect(html).toContain('\n\n');
	});

	it('produces the same body for framed and unframed input', () => {
		const trimmed = highlightCode('function foo() {}', 'javascript');
		const framed = highlightCode('\nfunction foo() {}\n', 'javascript');
		expect(framed).toBe(trimmed);
	});
});
