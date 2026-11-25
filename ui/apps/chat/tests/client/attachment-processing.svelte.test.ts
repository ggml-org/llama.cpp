import { AttachmentType } from '#lib/enums/index.ts';
import type { ChatUploadedFile } from '#lib/types/index.ts';
import { parseFilesToMessageExtras } from '#lib/utils/convert-files-to-extra.ts';
import { afterEach, describe, expect, it, vi } from 'vitest';

function upload(
	name: string,
	type: string,
	content: BlobPart = new Uint8Array([1, 2, 3])
): ChatUploadedFile {
	const file = new File([content], name, { type });

	return { file, id: name, name, size: file.size, type };
}

function failRead(name: string) {
	const read = FileReader.prototype.readAsDataURL;

	return vi.spyOn(FileReader.prototype, 'readAsDataURL').mockImplementation(function (
		this: FileReader,
		file: Blob
	) {
		if (file instanceof File && file.name === name) {
			Object.defineProperty(this, 'error', {
				value: new DOMException('The file is unavailable', 'NotReadableError')
			});
			queueMicrotask(() => this.dispatchEvent(new ProgressEvent('error')));

			return;
		}

		read.call(this, file);
	});
}

describe('attachment processing', () => {
	afterEach(() => vi.restoreAllMocks());

	it.each([
		{ name: 'clip.mp4', type: 'video/mp4' },
		{ name: 'speech.wav', type: 'audio/wav' }
	])('reports an unreadable $name and permits a retry', async ({ name, type }) => {
		const files = [
			upload('before.mp3', 'audio/mpeg'),
			upload(name, type),
			upload('after.mp3', 'audio/mpeg')
		];
		const failure = failRead(name);
		const failed = await parseFilesToMessageExtras(files);

		expect(failed.failedFiles).toEqual([name]);
		expect(failed.extras.map((extra) => extra.name)).toEqual(['before.mp3', 'after.mp3']);

		failure.mockRestore();

		const retry = await parseFilesToMessageExtras(files);

		expect(retry.failedFiles).toEqual([]);
		expect(retry.extras.map((extra) => extra.name)).toEqual(files.map((file) => file.name));
	});

	it('preserves audio and video bytes, metadata and order', async () => {
		const files = [upload('speech.wav', 'audio/wav'), upload('clip.mp4', 'video/mp4')];
		const result = await parseFilesToMessageExtras(files);

		expect(result.extras).toEqual([
			{
				base64Data: 'AQID',
				mimeType: 'audio/wav',
				name: 'speech.wav',
				size: 3,
				type: AttachmentType.AUDIO
			},
			{
				base64Data: 'AQID',
				mimeType: 'video/mp4',
				name: 'clip.mp4',
				size: 3,
				type: AttachmentType.VIDEO
			}
		]);
	});
	it.each([
		{ method: 'readAsDataURL', name: 'aborted.wav', type: 'audio/wav' },
		{ method: 'readAsText', name: 'aborted.txt', type: 'text/plain' }
	] as const)(
		'reports an aborted $name read and permits a retry',
		async ({ method, name, type }) => {
			const files = [upload(name, type, 'sample')];
			const abort = vi.spyOn(FileReader.prototype, method).mockImplementation(function (
				this: FileReader
			) {
				queueMicrotask(() => this.dispatchEvent(new ProgressEvent('abort')));
			});

			expect((await parseFilesToMessageExtras(files)).failedFiles).toEqual([name]);
			abort.mockRestore();
			expect((await parseFilesToMessageExtras(files)).extras[0].name).toBe(name);
		}
	);

	it('reports an unreadable PDF before conversion', async () => {
		const failure = failRead('document.pdf');
		const result = await parseFilesToMessageExtras([upload('document.pdf', 'application/pdf')]);

		expect(result.failedFiles).toEqual(['document.pdf']);
		expect(result.extras).toEqual([]);
		failure.mockRestore();
	});

	it('retries an image from the original file when no preview was saved', async () => {
		const files = [
			{ ...upload('good.png', 'image/png'), preview: 'data:image/png;base64,AQID' },
			upload('unreadable.png', 'image/png')
		];
		const failure = failRead('unreadable.png');
		const result = await parseFilesToMessageExtras(files);

		expect(result.failedFiles).toEqual(['unreadable.png']);
		expect(result.extras.map((extra) => extra.name)).toEqual(['good.png']);
		failure.mockRestore();
		const retry = await parseFilesToMessageExtras(files);

		expect(retry.failedFiles).toEqual([]);
		expect(retry.extras[1]).toMatchObject({
			base64Url: 'data:image/png;base64,AQID',
			name: 'unreadable.png'
		});
	});

	it('identifies the empty selection when two attachments share a filename', async () => {
		const name = 'notes.txt';
		const result = await parseFilesToMessageExtras(
			['', 'contents'].map((text, index) => ({
				file: new File([text], name, { type: 'text/plain' }),
				id: String(index),
				name,
				size: text.length,
				type: 'text/plain'
			}))
		);

		expect(result.emptyFiles).toEqual([{ id: '0', name }]);
		expect(result.extras).toEqual([expect.objectContaining({ content: 'contents', name })]);
	});

	it('reports a failed text read', async () => {
		vi.spyOn(FileReader.prototype, 'readAsText').mockImplementation(function (this: FileReader) {
			Object.defineProperty(this, 'error', {
				value: new DOMException('File unavailable', 'NotReadableError')
			});
			queueMicrotask(() => this.dispatchEvent(new ProgressEvent('error')));
		});
		const result = await parseFilesToMessageExtras([upload('notes.txt', 'text/plain')]);

		expect(result.failedFiles).toEqual(['notes.txt']);
		expect(result.extras).toEqual([]);
	});
});
