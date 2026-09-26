import { extractFilenameFromText } from '$lib/components/app/content/MarkdownContent/markdown-utils';
import { highlightCode, splitGluedClosingCodeFences, trimCodePadding } from '$lib/utils/code';
import { describe, expect, it } from 'vitest';

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

	it('auto-detects an unknown language by default', () => {
		const html = highlightCode('const answer = 42;', 'not-a-language');

		expect(html).toContain('hljs-');
	});

	it('escapes instead of auto-detecting when autoDetect is false', () => {
		const html = highlightCode('const answer = 42;', 'not-a-language', false);

		expect(html).not.toContain('hljs-');
		expect(html).toBe('const answer = 42;');
	});

	it('still highlights a known language when autoDetect is false', () => {
		const html = highlightCode('const answer = 42;', 'javascript', false);

		expect(html).toContain('hljs-');
	});

	it('escapes html metacharacters when falling back to plain text', () => {
		const html = highlightCode('<script>a && b</script>', 'not-a-language', false);

		expect(html).toBe('&lt;script&gt;a &amp;&amp; b&lt;/script&gt;');
	});
});

describe('splitGluedClosingCodeFences', () => {
	it('splits text glued to a closing fence onto its own line', () => {
		const input = "```ts\nlet foo = 'bar';\n```create this file on [Desktop](file:///a/b/)";

		expect(splitGluedClosingCodeFences(input)).toBe(
			"```ts\nlet foo = 'bar';\n```\ncreate this file on [Desktop](file:///a/b/)"
		);
	});

	it('leaves a well-formed code block untouched', () => {
		const input = "```ts\nlet foo = 'bar';\n```\ncreate this file on [Desktop](file:///a/b/)";

		expect(splitGluedClosingCodeFences(input)).toBe(input);
	});

	it('leaves content without fences untouched', () => {
		expect(splitGluedClosingCodeFences('hello world')).toBe('hello world');
	});

	it('keeps nested markdown fences inside a block intact', () => {
		const input = '```md\n# Example\n```python\nprint(1)\n```\n```';

		expect(splitGluedClosingCodeFences(input)).toBe(input);
	});

	it('splits every glued closing fence when several blocks are present', () => {
		const input = '```ts\na\n```first words\n\n```js\nb\n```second words';

		expect(splitGluedClosingCodeFences(input)).toBe(
			'```ts\na\n```\nfirst words\n\n```js\nb\n```\nsecond words'
		);
	});

	it('leaves a still-open fence untouched', () => {
		const input = '```ts\nlet foo = 1;';

		expect(splitGluedClosingCodeFences(input)).toBe(input);
	});
});

describe('extractFilenameFromText', () => {
	it('ignores unquoted wrongname and takes nearest quoted correct name', () => {
		const text =
			'These are the files to test the download button.\r\n' +
			'### 1. Ignore (**`wrongname.html`**) and take nearest quoted (**`correctname.html`**) not the unquoted wrongname.html\r\n' +
			'The download should save as the correct name .html\r\n' +
			'```';

		expect(extractFilenameFromText(text, text.lastIndexOf('```'), '.html')).toBe(
			'correctname.html'
		);
	});

	it('falls back to the last unquoted filename matching the target extension', () => {
		const text =
			'### 2. Unquoted filename wrongname.js wrongext.txt correct.name.js wrongext.md\r\n\r\n\r\n\r\n\r\n\r\n\r\n' +
			'This should fallback to the correct name as it requires to have the same extension as to the code block type.';

		expect(extractFilenameFromText(text, text.length, '.js')).toBe('correct.name.js');
	});

	it('strips virtual paths and trailing colons from unquoted matches', () => {
		const text =
			'### 3: Strip virtual path and colon suffix not from wrongpath/wrongname.md: but from wrongpath/correctname.md:\n' +
			'line2\nline3\n```markdown\n\n\n\n';

		expect(extractFilenameFromText(text, text.lastIndexOf('```'), '.md')).toBe('correctname.md');
	});

	it('type not specified in code block, accept quoted file name', () => {
		const text =
			'x'.repeat(5000) +
			'### 4. Code block type not specified. Here is **wrongname.js** sorry i mean (correct.name.js)\n' +
			'When quotes, trust the name and extension\n```\n';

		expect(extractFilenameFromText(text, text.lastIndexOf('```'), null)).toBe('correct.name.js');
	});

	it('type not specified in code block, do not accept unquoted file name', () => {
		const text =
			'this *wrongname.md* is too many lines behind\nx\nx\nx\nx\nx\n' +
			"### 5. Code block type not specified so don't accept a wrongname.ext\n" +
			'Not reliable enough that wrongname.txt or wrongname.md is the correct name as it is not quoted\n' +
			'```\n';

		expect(extractFilenameFromText(text, text.lastIndexOf('```'), '')).toBeNull();
	});

	it('returns null when no file matches the extension', () => {
		const text =
			'this *wrongname.md* is too far away ' +
			'x'.repeat(5000) +
			"### 6: Fallback to default (timestamp and .md) and ignore 'wrongext.txt' or 'wrongname .md'\n" +
			'This should save as llama_yyyymmdd_hhmmss .md';

		expect(extractFilenameFromText(text, text.length, '.md')).toBeNull();
	});

	it('rejects unquoted filenames with spaces around the extension dot', () => {
		const text =
			'// end of previous fence wrongname.ext\n```\n' +
			'### 7: Here is my  file . ext and also file .ext and another file. ext which are invalid.\n```';

		expect(extractFilenameFromText(text, text.lastIndexOf('```'), '.ext')).toBeNull();
	});

	it('rejects filenames with illegal OS characters', () => {
		const text =
			'### 8: Do take * correct name.txt *\n' +
			"and ignore **b@dname.txt** and 'worsename.123' or even **attack..traversal.txt**\n" +
			'also *-bad.x* *bad .x* *bad. x* *.nohidden.x* and look up to 3 lines back\n```text\n';

		expect(extractFilenameFromText(text, text.lastIndexOf('```'), '.txt')).toBe('correct name.txt');
	});
});
