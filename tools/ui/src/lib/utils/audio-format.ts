import { FileTypeAudio, MimeTypeAudio } from '$lib/enums';
import type { AudioInputFormat } from '$lib/types/api';

/**
 * Map a MIME type to the AudioInputFormat expected by the API.
 */
export function getAudioInputFormat(mimeType: string): AudioInputFormat {
	const normalizedMimeType = mimeType.trim().toLowerCase();

	if (
		normalizedMimeType === MimeTypeAudio.WAV ||
		normalizedMimeType === MimeTypeAudio.WAVE ||
		normalizedMimeType === MimeTypeAudio.X_WAV ||
		normalizedMimeType === MimeTypeAudio.X_WAVE ||
		normalizedMimeType === MimeTypeAudio.VND_WAVE ||
		normalizedMimeType === MimeTypeAudio.X_PN_WAV
	) {
		return FileTypeAudio.WAV;
	}

	return FileTypeAudio.MP3;
}
