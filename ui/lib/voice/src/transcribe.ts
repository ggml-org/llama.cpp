import {
	VOICE_API_PATHS,
	VOICE_CLIP_FILENAME,
	VOICE_CONTENT_TYPE_HEADER,
	VOICE_ERROR_MESSAGE,
	VOICE_HTML_BODY_PATTERN,
	VOICE_TRANSCRIPTION_FORM,
	VOICE_TRANSCRIPTION_RESPONSE_FORMAT
} from './constants.js';
import { VoiceTranscriptionError } from './errors.js';
import { cleanTranscript } from './transcript.js';
import type { TranscribeClipOptions, VoiceServerContext } from './types.js';

/**
 * Transcribes a recorded clip with `POST /v1/audio/transcriptions`: any model with
 * audio input can do it, the server turns it into a chat completion.
 * @returns the transcript, trimmed and without the model's own format (see cleanTranscript)
 * @throws VoiceTranscriptionError carrying the server's own message
 */
export async function transcribeClip(
	context: VoiceServerContext,
	blob: Blob,
	options: TranscribeClipOptions = {}
): Promise<string> {
	const form = new FormData();

	form.append(VOICE_TRANSCRIPTION_FORM.FILE, blob, options.filename ?? VOICE_CLIP_FILENAME.CLIP);

	if (options.model) form.append(VOICE_TRANSCRIPTION_FORM.MODEL, options.model);

	if (options.language) form.append(VOICE_TRANSCRIPTION_FORM.LANGUAGE, options.language);

	if (options.prompt) form.append(VOICE_TRANSCRIPTION_FORM.PROMPT, options.prompt);

	form.append(VOICE_TRANSCRIPTION_FORM.RESPONSE_FORMAT, VOICE_TRANSCRIPTION_RESPONSE_FORMAT);

	// no Content-Type: the browser sets the multipart boundary itself
	const headers = { ...context.getHeaders?.() };

	for (const key of Object.keys(headers)) {
		if (key.toLowerCase() === VOICE_CONTENT_TYPE_HEADER) delete headers[key];
	}

	const response = await (context.fetch ?? fetch)(
		context.resolveUrl(VOICE_API_PATHS.TRANSCRIPTIONS),
		{
			body: form,
			headers,
			method: 'POST',
			signal: options.signal
		}
	);

	if (!response.ok) {
		throw new VoiceTranscriptionError(await readErrorMessage(response), response.status);
	}

	const data = (await response.json()) as { text?: unknown };

	if (typeof data?.text !== 'string') {
		throw new VoiceTranscriptionError(VOICE_ERROR_MESSAGE.noTranscript, response.status);
	}

	return cleanTranscript(data.text).trim();
}

async function readErrorMessage(response: Response): Promise<string> {
	const fallback = `${VOICE_ERROR_MESSAGE.transcriptionFailed} (${response.status}${response.statusText ? ` ${response.statusText}` : ''})`;

	try {
		const text = await response.text();

		try {
			const data = JSON.parse(text);

			if (typeof data?.error?.message === 'string') return data.error.message;

			if (typeof data?.error === 'string') return data.error;

			if (typeof data?.message === 'string') return data.message;
		} catch {
			// not JSON: a proxy's plain text error
		}

		const plain = text.trim();

		// an HTML error page says nothing useful in a toast
		return plain && !VOICE_HTML_BODY_PATTERN.test(plain) ? plain : fallback;
	} catch {
		return fallback;
	}
}
