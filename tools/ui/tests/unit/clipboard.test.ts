import { AttachmentType } from '$lib/enums';
import {
	formatMessageForClipboard,
	hasClipboardAttachments,
	parseClipboardContent
} from '$lib/utils/clipboard';
import { describe, expect, it } from 'vitest';

describe('formatMessageForClipboard', () => {
	it('returns plain content when no extras', () => {
		const result = formatMessageForClipboard('Hello world', undefined);

		expect(result).toBe('Hello world');
	});

	it('returns plain content when extras is empty array', () => {
		const result = formatMessageForClipboard('Hello world', []);

		expect(result).toBe('Hello world');
	});

	it('handles empty string content', () => {
		const result = formatMessageForClipboard('', undefined);

		expect(result).toBe('');
	});

	it('returns plain content when extras has only non-text attachments', () => {
		const extras = [
			{
				base64Url: 'data:image/png;base64,...',
				name: 'image.png',
				type: AttachmentType.IMAGE as const
			}
		];
		const result = formatMessageForClipboard('Hello world', extras);

		expect(result).toBe('Hello world');
	});

	it('filters non-text attachments and keeps only text ones', () => {
		const extras = [
			{
				base64Url: 'data:image/png;base64,...',
				name: 'image.png',
				type: AttachmentType.IMAGE as const
			},
			{
				content: 'Text content',
				name: 'file.txt',
				type: AttachmentType.TEXT as const
			},
			{
				base64Data: 'data:application/pdf;base64,...',
				content: 'PDF content',
				name: 'doc.pdf',
				processedAsImages: false,
				type: AttachmentType.PDF as const
			}
		];
		const result = formatMessageForClipboard('Hello', extras);

		expect(result).toContain('"file.txt"');
		expect(result).not.toContain('image.png');
		expect(result).not.toContain('doc.pdf');
	});

	it('formats message with text attachments', () => {
		const extras = [
			{
				content: 'File 1 content',
				name: 'file1.txt',
				type: AttachmentType.TEXT as const
			},
			{
				content: 'File 2 content',
				name: 'file2.txt',
				type: AttachmentType.TEXT as const
			}
		];
		const result = formatMessageForClipboard('Hello world', extras);

		expect(result).toContain('"Hello world"');
		expect(result).toContain('"type": "TEXT"');
		expect(result).toContain('"name": "file1.txt"');
		expect(result).toContain('"content": "File 1 content"');
		expect(result).toContain('"name": "file2.txt"');
	});

	it('handles content with quotes and special characters', () => {
		const content = 'Hello "world" with\nnewline';
		const extras = [
			{
				content: 'Test content',
				name: 'test.txt',
				type: AttachmentType.TEXT as const
			}
		];
		const result = formatMessageForClipboard(content, extras);

		// Should be valid JSON
		expect(result.startsWith('"')).toBe(true);
		// The content should be properly escaped
		const parsed = JSON.parse(result.split('\n')[0]);

		expect(parsed).toBe(content);
	});

	it('converts legacy context type to TEXT type', () => {
		const extras = [
			{
				content: 'Legacy content',
				name: 'legacy.txt',
				type: AttachmentType.LEGACY_CONTEXT as const
			}
		];
		const result = formatMessageForClipboard('Hello', extras);

		expect(result).toContain('"type": "TEXT"');
		expect(result).not.toContain('"context"');
	});

	it('handles attachment content with special characters', () => {
		const extras = [
			{
				content: 'const x = "hello\\nworld";\nconst y = `template ${var}`;',
				name: 'code.js',
				type: AttachmentType.TEXT as const
			}
		];
		const formatted = formatMessageForClipboard('Check this code', extras);
		const parsed = parseClipboardContent(formatted);

		expect(parsed.textAttachments[0].content).toBe(
			'const x = "hello\\nworld";\nconst y = `template ${var}`;'
		);
	});

	it('handles unicode characters in content and attachments', () => {
		const extras = [
			{
				content: '日本語テスト 🎉 émojis',
				name: 'unicode.txt',
				type: AttachmentType.TEXT as const
			}
		];
		const formatted = formatMessageForClipboard('Привет мир 👋', extras);
		const parsed = parseClipboardContent(formatted);

		expect(parsed.message).toBe('Привет мир 👋');
		expect(parsed.textAttachments[0].content).toBe('日本語テスト 🎉 émojis');
	});

	it('formats as plain text when asPlainText is true', () => {
		const extras = [
			{
				content: 'File 1 content',
				name: 'file1.txt',
				type: AttachmentType.TEXT as const
			},
			{
				content: 'File 2 content',
				name: 'file2.txt',
				type: AttachmentType.TEXT as const
			}
		];
		const result = formatMessageForClipboard('Hello world', extras, true);

		expect(result).toBe('Hello world\n\nFile 1 content\n\nFile 2 content');
	});

	it('returns plain content when asPlainText is true but no attachments', () => {
		const result = formatMessageForClipboard('Hello world', [], true);

		expect(result).toBe('Hello world');
	});

	it('plain text mode does not use JSON format', () => {
		const extras = [
			{
				content: 'Test content',
				name: 'test.txt',
				type: AttachmentType.TEXT as const
			}
		];
		const result = formatMessageForClipboard('Hello', extras, true);

		expect(result).not.toContain('"type"');
		expect(result).not.toContain('[');
		expect(result).toBe('Hello\n\nTest content');
	});
});

describe('parseClipboardContent', () => {
	it('returns plain text as message when not in special format', () => {
		const result = parseClipboardContent('Hello world');

		expect(result.message).toBe('Hello world');
		expect(result.textAttachments).toHaveLength(0);
	});

	it('handles empty string input', () => {
		const result = parseClipboardContent('');

		expect(result.message).toBe('');
		expect(result.textAttachments).toHaveLength(0);
	});

	it('handles whitespace-only input', () => {
		const result = parseClipboardContent('   \n\t  ');

		expect(result.message).toBe('   \n\t  ');
		expect(result.textAttachments).toHaveLength(0);
	});

	it('returns plain text as message when starts with quote but invalid format', () => {
		const result = parseClipboardContent('"Unclosed quote');

		expect(result.message).toBe('"Unclosed quote');
		expect(result.textAttachments).toHaveLength(0);
	});

	it('returns original text when JSON array is malformed', () => {
		const input = '"Hello"\n[invalid json';
		const result = parseClipboardContent(input);

		expect(result.message).toBe('"Hello"\n[invalid json');
		expect(result.textAttachments).toHaveLength(0);
	});

	it('parses message with text attachments', () => {
		const input = `"Hello world"
[
  {"type":"TEXT","name":"file1.txt","content":"File 1 content"},
  {"type":"TEXT","name":"file2.txt","content":"File 2 content"}
]`;
		const result = parseClipboardContent(input);

		expect(result.message).toBe('Hello world');
		expect(result.textAttachments).toHaveLength(2);
		expect(result.textAttachments[0].name).toBe('file1.txt');
		expect(result.textAttachments[0].content).toBe('File 1 content');
		expect(result.textAttachments[1].name).toBe('file2.txt');
		expect(result.textAttachments[1].content).toBe('File 2 content');
	});

	it('handles escaped quotes in message', () => {
		const input = `"Hello \\"world\\" with quotes"
[
  {"type":"TEXT","name":"file.txt","content":"test"}
]`;
		const result = parseClipboardContent(input);

		expect(result.message).toBe('Hello "world" with quotes');
		expect(result.textAttachments).toHaveLength(1);
	});

	it('handles newlines in message', () => {
		const input = `"Hello\\nworld"
[
  {"type":"TEXT","name":"file.txt","content":"test"}
]`;
		const result = parseClipboardContent(input);

		expect(result.message).toBe('Hello\nworld');
		expect(result.textAttachments).toHaveLength(1);
	});

	it('returns message only when no array follows', () => {
		const input = '"Just a quoted string"';
		const result = parseClipboardContent(input);

		expect(result.message).toBe('Just a quoted string');
		expect(result.textAttachments).toHaveLength(0);
	});

	it('filters out invalid attachment objects', () => {
		const input = `"Hello"
[
  {"type":"TEXT","name":"valid.txt","content":"valid"},
  {"type":"INVALID","name":"invalid.txt","content":"invalid"},
  {"name":"missing-type.txt","content":"missing"},
  {"type":"TEXT","content":"missing name"}
]`;
		const result = parseClipboardContent(input);

		expect(result.message).toBe('Hello');
		expect(result.textAttachments).toHaveLength(1);
		expect(result.textAttachments[0].name).toBe('valid.txt');
	});

	it('handles empty attachments array', () => {
		const input = '"Hello"\n[]';
		const result = parseClipboardContent(input);

		expect(result.message).toBe('Hello');
		expect(result.textAttachments).toHaveLength(0);
	});

	it('roundtrips correctly with formatMessageForClipboard', () => {
		const originalContent = 'Hello "world" with\nspecial characters';
		const originalExtras = [
			{
				content: 'Content with\nnewlines and "quotes"',
				name: 'file1.txt',
				type: AttachmentType.TEXT as const
			},
			{
				content: 'Another file',
				name: 'file2.txt',
				type: AttachmentType.TEXT as const
			}
		];
		const formatted = formatMessageForClipboard(originalContent, originalExtras);
		const parsed = parseClipboardContent(formatted);

		expect(parsed.message).toBe(originalContent);
		expect(parsed.textAttachments).toHaveLength(2);
		expect(parsed.textAttachments[0].name).toBe('file1.txt');
		expect(parsed.textAttachments[0].content).toBe('Content with\nnewlines and "quotes"');
		expect(parsed.textAttachments[1].name).toBe('file2.txt');
		expect(parsed.textAttachments[1].content).toBe('Another file');
	});
});

describe('hasClipboardAttachments', () => {
	it('returns false for plain text', () => {
		expect(hasClipboardAttachments('Hello world')).toBe(false);
	});

	it('returns false for empty string', () => {
		expect(hasClipboardAttachments('')).toBe(false);
	});

	it('returns false for quoted string without attachments', () => {
		expect(hasClipboardAttachments('"Hello world"')).toBe(false);
	});

	it('returns true for valid format with attachments', () => {
		const input = `"Hello"
[{"type":"TEXT","name":"file.txt","content":"test"}]`;

		expect(hasClipboardAttachments(input)).toBe(true);
	});

	it('returns false for format with empty attachments array', () => {
		const input = '"Hello"\n[]';

		expect(hasClipboardAttachments(input)).toBe(false);
	});

	it('returns false for malformed JSON', () => {
		expect(hasClipboardAttachments('"Hello"\n[broken')).toBe(false);
	});
});

describe('roundtrip edge cases', () => {
	it('preserves empty message with attachments', () => {
		const extras = [
			{
				content: 'Content only',
				name: 'file.txt',
				type: AttachmentType.TEXT as const
			}
		];
		const formatted = formatMessageForClipboard('', extras);
		const parsed = parseClipboardContent(formatted);

		expect(parsed.message).toBe('');
		expect(parsed.textAttachments).toHaveLength(1);
		expect(parsed.textAttachments[0].content).toBe('Content only');
	});

	it('preserves attachment with empty content', () => {
		const extras = [
			{
				content: '',
				name: 'empty.txt',
				type: AttachmentType.TEXT as const
			}
		];
		const formatted = formatMessageForClipboard('Message', extras);
		const parsed = parseClipboardContent(formatted);

		expect(parsed.message).toBe('Message');
		expect(parsed.textAttachments).toHaveLength(1);
		expect(parsed.textAttachments[0].content).toBe('');
	});

	it('preserves multiple backslashes', () => {
		const content = 'Path: C:\\\\Users\\\\test\\\\file.txt';
		const extras = [
			{
				content: 'D:\\\\Data\\\\file',
				name: 'path.txt',
				type: AttachmentType.TEXT as const
			}
		];
		const formatted = formatMessageForClipboard(content, extras);
		const parsed = parseClipboardContent(formatted);

		expect(parsed.message).toBe(content);
		expect(parsed.textAttachments[0].content).toBe('D:\\\\Data\\\\file');
	});

	it('preserves tabs and various whitespace', () => {
		const content = 'Line1\t\tTabbed\n  Spaced\r\nCRLF';
		const extras = [
			{
				content: '\t\t\n\n   ',
				name: 'whitespace.txt',
				type: AttachmentType.TEXT as const
			}
		];
		const formatted = formatMessageForClipboard(content, extras);
		const parsed = parseClipboardContent(formatted);

		expect(parsed.message).toBe(content);
		expect(parsed.textAttachments[0].content).toBe('\t\t\n\n   ');
	});
});
