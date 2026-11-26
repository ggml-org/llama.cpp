import { MimeTypeAudio, MimeTypeImage, MimeTypeVideo } from '#lib/enums/index.ts';
import { getMediaMimeTypeByExtension } from '#lib/utils/index.ts';
import { describe, expect, it } from 'vitest';

describe('getMediaMimeTypeByExtension', () => {
	it('tells an image, an audio clip and a video by their extension', () => {
		expect(getMediaMimeTypeByExtension('/home/al/Desktop/Screenshot 1.PNG')).toBe(
			MimeTypeImage.PNG
		);
		expect(getMediaMimeTypeByExtension('voice.mp3')).toBe(MimeTypeAudio.MP3_MPEG);
		expect(getMediaMimeTypeByExtension('clip.webm')).toBe(MimeTypeVideo.WEBM);
	});

	it('returns null for anything that is not media', () => {
		expect(getMediaMimeTypeByExtension('notes.md')).toBeNull();
		expect(getMediaMimeTypeByExtension('report.pdf')).toBeNull();
		expect(getMediaMimeTypeByExtension('Makefile')).toBeNull();
	});
});
