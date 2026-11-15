import { VOICE_API_PATHS } from './constants';
import { VoiceTranscriptionError } from './errors';
import type { TranscribeClipOptions, VoiceServerContext } from './types';

/**
 * Transcribes a recorded clip with `POST /v1/audio/transcriptions`: any model with
 * audio input can do it, the server turns it into a chat completion.
 * @returns the transcript, trimmed
 * @throws VoiceTranscriptionError carrying the server's own message
 */
export async function transcribeClip(
	context: VoiceServerContext,
	blob: Blob,
	options: TranscribeClipOptions = {}
): Promise<string> {
	const form = new FormData();

	form.append('file', blob, options.filename ?? 'audio.wav');

	if (options.model) form.append('model', options.model);

	if (options.language) form.append('language', options.language);

	if (options.prompt) form.append('prompt', options.prompt);

	form.append('response_format', 'json');

	// no Content-Type: the browser sets the multipart boundary itself
	const headers = { ...context.getHeaders?.() };

	for (const key of Object.keys(headers)) {
		if (key.toLowerCase() === 'content-type') delete headers[key];
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
		throw new VoiceTranscriptionError('The server returned no transcript.', response.status);
	}

	return data.text.trim();
}

async function readErrorMessage(response: Response): Promise<string> {
	const fallback = `Transcription failed (${response.status}${response.statusText ? ` ${response.statusText}` : ''})`;

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
		return plain && !plain.startsWith('<') ? plain : fallback;
	} catch {
		return fallback;
	}
}
