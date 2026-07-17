import { describe, it, expect } from 'vitest';
import { classifyToolResult } from '$lib/utils/agentic';

describe('classifyToolResult', () => {
	describe('text', () => {
		it('returns text for undefined input', () => {
			expect(classifyToolResult(undefined)).toBe('text');
		});

		it('returns text for empty string', () => {
			expect(classifyToolResult('')).toBe('text');
		});

		it('returns text for whitespace-only input', () => {
			expect(classifyToolResult('   \n  ')).toBe('text');
		});

		it('returns text for plain prose', () => {
			expect(classifyToolResult('Hello, this is just some text.')).toBe('text');
		});

		it('returns text for shell-style line listings', () => {
			expect(classifyToolResult('file1.java\nfile2.java\nfile3.java\n')).toBe('text');
		});

		it('returns text when a brace-like string is not valid JSON', () => {
			expect(classifyToolResult('{key: value}')).toBe('text');
		});
	});

	describe('json', () => {
		it('classifies a flat JSON object', () => {
			expect(classifyToolResult('{"key": "value", "n": 42}')).toBe('json');
		});

		it('classifies a JSON array', () => {
			expect(classifyToolResult('["a", "b", "c"]')).toBe('json');
		});

		it('classifies a pretty-printed JSON object', () => {
			expect(classifyToolResult('{\n  "key": "value"\n}')).toBe('json');
		});

		it('classifies a deeply nested JSON payload', () => {
			const nested = JSON.stringify({ items: [{ id: 1, tags: ['a', 'b'] }] }, null, 2);
			expect(classifyToolResult(nested)).toBe('json');
		});

		it('prefers JSON over inner markdown markers when the content starts with a brace', () => {
			// A JSON object whose inner strings contain link syntax still
			// reads as JSON because the leading `{` parses cleanly -
			// `classifyToolResult` only inspects the top-level shape, not
			// every nested line marker.
			const jsonWithLink = '{"docs": "see [docs](https://example.com) for more"}';
			expect(classifyToolResult(jsonWithLink)).toBe('json');
		});
	});

	describe('markdown', () => {
		it('classifies an ATX header line', () => {
			expect(classifyToolResult('# Title\n\nSome text below.')).toBe('markdown');
		});

		it('classifies a fenced code block', () => {
			expect(classifyToolResult('```json\n{"key": "v"}\n```')).toBe('markdown');
		});

		it('classifies a tilde-fenced code block', () => {
			expect(classifyToolResult('~~~bash\nls -la\n~~~')).toBe('markdown');
		});

		it('classifies a markdown link', () => {
			expect(classifyToolResult('See [docs](https://example.com) for more.')).toBe('markdown');
		});

		it('classifies bold text', () => {
			expect(classifyToolResult('This is **very important**.')).toBe('markdown');
		});

		it('classifies a bulleted list', () => {
			expect(classifyToolResult('- item one\n- item two\n- item three')).toBe('markdown');
		});

		it('classifies an ordered list', () => {
			expect(classifyToolResult('1. first step\n2. second step\n3. third step')).toBe('markdown');
		});

		it('classifies a blockquote', () => {
			expect(classifyToolResult('> quoted text\n> second line')).toBe('markdown');
		});

		it('classifies a markdown table', () => {
			const table = '| a | b |\n| - | - |\n| 1 | 2 |';
			expect(classifyToolResult(table)).toBe('markdown');
		});

		it('classifies a markdown table with alignment markers', () => {
			const table = '| left | center | right |\n| :--- | :---: | ---: |\n| a | b | c |';
			expect(classifyToolResult(table)).toBe('markdown');
		});

		it('classifies nested markdown headings', () => {
			expect(classifyToolResult('## Section\n\n### Subsection\n')).toBe('markdown');
		});

		it('classifies combined markdown markers in one document', () => {
			const md = [
				'# Heading',
				'',
				'A paragraph with a [link](https://example.com) and **bold text**.',
				'',
				'- bullet item',
				'- another bullet',
				'',
				'| col1 | col2 |',
				'| ----- | ----- |',
				'| a     | b     |'
			].join('\n');
			expect(classifyToolResult(md)).toBe('markdown');
		});
	});

	describe('precedence', () => {
		it('prefers JSON over markdown when both signals are present', () => {
			// Starts with `[`, parses as JSON - markdown check is skipped.
			const arr = '[1, 2, "# not-a-heading", "**not-bold**"]';
			expect(classifyToolResult(arr)).toBe('json');
		});
	});
});
