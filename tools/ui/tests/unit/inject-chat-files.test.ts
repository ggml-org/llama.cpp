import { AttachmentType, MessageRole } from '$lib/enums';
import type { DatabaseMessageExtra } from '$lib/types/database';
import {
	collectLastUserMessageExtras,
	extrasToChatFiles,
	injectChatFilesIntoToolArgs
} from '$lib/utils/inject-chat-files';
import { describe, expect, it } from 'vitest';

const pngDataUrl = 'data:image/png;base64,iVBORw0KGgo=';
const pdfMime = 'application/pdf';
const pdfDataUrl = `data:${pdfMime};base64,JVBERi0=`;

describe('injectChatFilesIntoToolArgs', () => {
	it('injects __files__, __file__, __image__, image, and file_id for a PNG', () => {
		const extras: DatabaseMessageExtra[] = [
			{ base64Url: pngDataUrl, name: 'scan.png', type: AttachmentType.IMAGE }
		];
		const out = injectChatFilesIntoToolArgs({}, extras);

		expect(out.__image__).toBe(pngDataUrl);
		expect(out.image).toBe(pngDataUrl);
		expect(out.file_id).toBe('scan.png');
		expect(out.__file__).toMatchObject({ name: 'scan.png', type: 'image', url: pngDataUrl });
		expect(out.__files__).toHaveLength(1);
	});

	it('injects a PDF as type file with a data URI', () => {
		const extras: DatabaseMessageExtra[] = [
			{
				base64Data: 'JVBERi0=',
				content: '',
				name: 'doc.pdf',
				processedAsImages: false,
				type: AttachmentType.PDF
			}
		];
		const out = injectChatFilesIntoToolArgs({ task: 'v1.5' }, extras);

		expect(out.image).toBe('data:application/pdf;base64,JVBERi0=');
		expect(out.__file__).toMatchObject({
			mimeType: 'application/pdf',
			name: 'doc.pdf',
			type: 'file',
			url: pdfDataUrl
		});
		expect(out.task).toBe('v1.5');
	});

	it('does not overwrite image or __files__ the model already set', () => {
		const extras: DatabaseMessageExtra[] = [
			{ base64Url: pngDataUrl, name: 'scan.png', type: AttachmentType.IMAGE }
		];
		const out = injectChatFilesIntoToolArgs(
			{ __files__: ['keep'], image: '/tmp/explicit.png' },
			extras
		);

		expect(out.image).toBe('/tmp/explicit.png');
		expect(out.__files__).toEqual(['keep']);
		expect(out.__image__).toBe(pngDataUrl);
	});

	it('returns args unchanged when there are no file extras', () => {
		const extras: DatabaseMessageExtra[] = [
			{ content: 'hello', name: 'notes.txt', type: AttachmentType.TEXT }
		];

		expect(injectChatFilesIntoToolArgs({ page: '1' }, extras)).toEqual({ page: '1' });
	});
});

describe('collectLastUserMessageExtras', () => {
	it('returns extras from the last user message', () => {
		const extras = collectLastUserMessageExtras([
			{
				extra: [{ base64Url: pngDataUrl, name: 'old.png', type: AttachmentType.IMAGE }],
				role: MessageRole.USER
			},
			{ extra: [], role: MessageRole.ASSISTANT },
			{
				extra: [
					{
						base64Data: 'JVBERi0=',
						content: '',
						name: 'latest.pdf',
						processedAsImages: false,
						type: AttachmentType.PDF
					}
				],
				role: MessageRole.USER
			}
		]);

		expect(extras).toHaveLength(1);
		expect(extras[0]).toMatchObject({ name: 'latest.pdf' });
	});
});

describe('extrasToChatFiles', () => {
	it('maps image and pdf extras and skips text', () => {
		const files = extrasToChatFiles([
			{ base64Url: pngDataUrl, name: 'a.png', type: AttachmentType.IMAGE },
			{ content: 'x', name: 'a.txt', type: AttachmentType.TEXT },
			{
				base64Data: 'JVBERi0=',
				content: '',
				name: 'a.pdf',
				processedAsImages: false,
				type: AttachmentType.PDF
			}
		]);

		expect(files.map((f) => f.type)).toEqual(['image', 'file']);
		expect(files[1].url).toBe(pdfDataUrl);
	});
});
