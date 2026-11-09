import { AttachmentType } from '$lib/enums';
import type { ChatUploadedFile } from '$lib/types';
import { parseFilesToMessageExtras } from '$lib/utils/convert-files-to-extra';
import { afterEach, describe, expect, it, vi } from 'vitest';

function upload(name: string, type: string): ChatUploadedFile {
	const file = new File([new Uint8Array([1, 2, 3])], name, { type });

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
	it('reports an aborted media read and permits a retry', async () => {
		const files = [upload('aborted.wav', 'audio/wav')];
		const abort = vi.spyOn(FileReader.prototype, 'readAsDataURL').mockImplementation(function (
			this: FileReader
		) {
			queueMicrotask(() => this.dispatchEvent(new ProgressEvent('abort')));
		});

		expect((await parseFilesToMessageExtras(files)).failedFiles).toEqual(['aborted.wav']);
		abort.mockRestore();
		expect((await parseFilesToMessageExtras(files)).extras[0].name).toBe('aborted.wav');
	});

	it('reports an unreadable PDF before conversion', async () => {
		const failure = failRead('document.pdf');
		const result = await parseFilesToMessageExtras([upload('document.pdf', 'application/pdf')]);

		expect(result.failedFiles).toEqual(['document.pdf']);
		expect(result.extras).toEqual([]);
		failure.mockRestore();
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
