import { base64ToBytes, computePeaksFromAudio } from '@ggml-org/llama-voice';
import {
	VOICE_FETCHABLE_URL_PATTERN,
	VOICE_PEAKS_CACHE_TAIL_LENGTH,
	VOICE_URL_PATTERN
} from '#lib/constants/index.ts';
import { AttachmentType, MimeTypeAudio } from '#lib/enums/index.ts';
import type { ChatAttachmentDisplayItem, DatabaseMessageExtraAudioFile } from '#lib/types/index.ts';

// peaks of clips stored without them, kept for the session only
const peaksCache = new Map<string, Promise<number[]>>();

/** What a list item needs to draw an audio attachment as a clip. */
export interface VoiceAttachmentSource {
	/** Base64 or a URL of the audio, or the file itself. */
	audio: string | Blob | null;
	durationMs?: number;
	mimeType: string;
	name: string;
	peaks?: number[];
	/** Recorded in the app, as opposed to an uploaded file. */
	voice: boolean;
	transcript?: string;
}

/** The clip behind an audio item of the form or of a message. */
export function voiceAttachmentSource(item: ChatAttachmentDisplayItem): VoiceAttachmentSource {
	const uploaded = item.uploadedFile;

	if (uploaded) {
		return {
			audio: uploaded.preview ?? (uploaded.file.size > 0 ? uploaded.file : null),
			durationMs: uploaded.voice?.durationMs,
			mimeType: uploaded.type,
			name: uploaded.name,
			peaks: uploaded.voice?.peaks,
			voice: !!uploaded.voice
		};
	}

	const attachment =
		item.attachment?.type === AttachmentType.AUDIO
			? (item.attachment as DatabaseMessageExtraAudioFile)
			: null;

	return {
		audio: attachment?.base64Data || null,
		durationMs: attachment?.durationMs,
		mimeType: attachment?.mimeType ?? '',
		name: item.name,
		peaks: attachment?.peaks,
		transcript: attachment?.transcript,
		voice: attachment?.voice === true
	};
}

/** A URL an <audio> can play: URLs as they are, base64 and files as object URLs; null for bad data. */
export function audioUrl(
	audio: string | Blob,
	mimeType: string
): { url: string; revoke: boolean } | null {
	// a data, blob or http URL plays as it is
	if (typeof audio === 'string' && VOICE_URL_PATTERN.test(audio)) {
		return { revoke: false, url: audio };
	}

	if (typeof audio !== 'string') return { revoke: true, url: URL.createObjectURL(audio) };

	let bytes: Uint8Array<ArrayBuffer>;

	try {
		bytes = base64ToBytes(audio);
	} catch {
		// not base64: nothing to play
		return null;
	}

	return {
		revoke: true,
		url: URL.createObjectURL(new Blob([bytes], { type: mimeType || MimeTypeAudio.WAV }))
	};
}

/** Peaks of a clip stored without them, computed once per session. */
export function lazyPeaks(name: string, audio: string | Blob): Promise<number[]> {
	const key =
		typeof audio === 'string'
			? `${name}:${audio.length}:${audio.slice(-VOICE_PEAKS_CACHE_TAIL_LENGTH)}`
			: `${name}:${audio.size}`;

	let peaks = peaksCache.get(key);

	if (!peaks) {
		const isUrl = typeof audio === 'string' && VOICE_FETCHABLE_URL_PATTERN.test(audio);
		const bytes = isUrl ? fetch(audio as string).then((r) => r.blob()) : Promise.resolve(audio);

		peaks = bytes
			.then((source) => computePeaksFromAudio(source))
			.catch((error) => {
				console.warn(`Could not draw the waveform of ${name}:`, error);
				peaksCache.delete(key);

				return [];
			});
		peaksCache.set(key, peaks);
	}

	return peaks;
}
