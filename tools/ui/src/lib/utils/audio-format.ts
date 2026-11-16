import { FileTypeAudio, MimeTypeAudio } from '$lib/enums';
import type { AudioInputFormat } from '$lib/types/api';

const WAV_MIME_TYPES: string[] = [
	MimeTypeAudio.WAV,
	MimeTypeAudio.WAVE,
	MimeTypeAudio.X_WAV,
	MimeTypeAudio.X_WAVE,
	MimeTypeAudio.VND_WAVE,
	MimeTypeAudio.X_PN_WAV
];
const MP3_MIME_TYPES: string[] = [MimeTypeAudio.MP3, MimeTypeAudio.MP3_MPEG];

function isWavMimeType(mimeType: string): boolean {
	return WAV_MIME_TYPES.includes(mimeType.trim().toLowerCase());
}

/**
 * Check if the server can decode this audio MIME type as is (WAV or MP3).
 */
export function isServerDecodableAudio(mimeType: string): boolean {
	return isWavMimeType(mimeType) || MP3_MIME_TYPES.includes(mimeType.trim().toLowerCase());
}

/**
 * Map a MIME type to the AudioInputFormat expected by the API.
 */
export function getAudioInputFormat(mimeType: string): AudioInputFormat {
	if (isWavMimeType(mimeType)) {
		return FileTypeAudio.WAV;
	}

	return FileTypeAudio.MP3;
}
