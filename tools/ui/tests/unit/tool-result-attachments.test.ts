import { describe, expect, it, vi } from 'vitest';
import { AttachmentType } from '$lib/enums';
import { extractToolResultAttachments } from '$lib/utils/tool-result-attachments';

describe('extractToolResultAttachments', () => {
	it('extracts text attachments from data URIs', () => {
		vi.spyOn(Date, 'now').mockReturnValue(1710000000000);

		const result = extractToolResultAttachments(
			'before\ndata:text/markdown;base64,IyBIZWxsbyBXb3JsZA==\nafter'
		);

		expect(result.cleanedResult).toContain('[Attachment saved: mcp-attachment-1710000000000-1.md]');
		expect(result.attachments).toEqual([
			{
				type: AttachmentType.TEXT,
				name: 'mcp-attachment-1710000000000-1.md',
				content: '# Hello World',
				size: 13
			}
		]);
	});

	it('extracts pdf, audio, and video attachments', () => {
		vi.spyOn(Date, 'now').mockReturnValue(1710000000001);

		const pdf = 'data:application/pdf;base64,JVBERi0xLjQ=';
		const audio = 'data:audio/mpeg;base64,QUJDRA==';
		const video = 'data:video/mp4;base64,QUJDRA==';
		const result = extractToolResultAttachments([pdf, audio, video].join('\n'));

		expect(result.cleanedResult).toContain('mcp-attachment-1710000000001-1.pdf');
		expect(result.cleanedResult).toContain('mcp-attachment-1710000000001-2.mp3');
		expect(result.cleanedResult).toContain('mcp-attachment-1710000000001-3.mp4');
		expect(result.attachments).toEqual([
			{
				type: AttachmentType.PDF,
				name: 'mcp-attachment-1710000000001-1.pdf',
				base64Data: 'JVBERi0xLjQ=',
				content: '',
				processedAsImages: false,
				size: 8
			},
			{
				type: AttachmentType.AUDIO,
				name: 'mcp-attachment-1710000000001-2.mp3',
				mimeType: 'audio/mpeg',
				base64Data: 'QUJDRA==',
				size: 4
			},
			{
				type: AttachmentType.VIDEO,
				name: 'mcp-attachment-1710000000001-3.mp4',
				mimeType: 'video/mp4',
				base64Data: 'QUJDRA==',
				size: 4
			}
		]);
	});

	it('leaves unsupported or invalid lines unchanged', () => {
		const original = 'hello\ndata:application/octet-stream;base64,QUJDRA==\nworld';
		const result = extractToolResultAttachments(original);

		expect(result.cleanedResult).toBe(original);
		expect(result.attachments).toEqual([]);
	});
});
