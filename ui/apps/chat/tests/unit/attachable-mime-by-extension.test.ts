import { MimeTypeApplication, MimeTypeAudio, MimeTypeImage, MimeTypeVideo } from '#lib/enums/index.ts';
import { getAttachableMimeTypeByExtension } from '#lib/utils/index.ts';
import { describe, expect, it } from 'vitest';

describe('getAttachableMimeTypeByExtension', () => {
	it('tells an image, an audio clip, a video and a PDF by their extension', () => {
		expect(getAttachableMimeTypeByExtension('/home/al/Desktop/Screenshot 1.PNG')).toBe(
			MimeTypeImage.PNG
		);
		expect(getAttachableMimeTypeByExtension('voice.mp3')).toBe(MimeTypeAudio.MP3_MPEG);
		expect(getAttachableMimeTypeByExtension('clip.webm')).toBe(MimeTypeVideo.WEBM);
		expect(getAttachableMimeTypeByExtension('Screen Recording 1.mov')).toBe(MimeTypeVideo.QUICKTIME);
		expect(getAttachableMimeTypeByExtension('~/Documents/Report.PDF')).toBe(MimeTypeApplication.PDF);
	});

	it('returns null for anything read as text or not attachable', () => {
		expect(getAttachableMimeTypeByExtension('notes.md')).toBeNull();
		expect(getAttachableMimeTypeByExtension('Makefile')).toBeNull();
	});
});
