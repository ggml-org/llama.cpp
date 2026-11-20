// Guards the HTTP transcription client: the multipart request
// /v1/audio/transcriptions expects, auth headers, and the server's error messages.

import { VoiceTranscriptionError } from '../src/errors.js';
import { transcribeClip } from '../src/transcribe.js';
import { type VoiceServerContext } from '../src/types.js';
import { describe, expect, it, vi } from 'vitest';

function contextWith(fetchImpl: typeof fetch): VoiceServerContext {
	return {
		fetch: fetchImpl,
		getHeaders: () => ({ Authorization: 'Bearer secret', 'Content-Type': 'application/json' }),
		resolveUrl: (path) => `http://server${path.replace(/^\./, '')}`
	};
}

describe('transcribeClip', () => {
	it('posts the clip as multipart with the model and json format', async () => {
		const fetchMock = vi.fn(
			async (_url: RequestInfo | URL, _init?: RequestInit) =>
				new Response(JSON.stringify({ text: ' Hello there. ' }), { status: 200 })
		);
		const blob = new Blob([new Uint8Array([1, 2, 3])], { type: 'audio/wav' });
		const text = await transcribeClip(contextWith(fetchMock as typeof fetch), blob, {
			language: 'en',
			model: 'org/model:Q8_0'
		});

		expect(text).toBe('Hello there.');

		const [url, init] = fetchMock.mock.calls[0];
		const form = init!.body as FormData;
		const file = form.get('file') as File;

		expect(url).toBe('http://server/v1/audio/transcriptions');
		expect(init!.method).toBe('POST');
		// the browser sets the multipart Content-Type with its boundary
		expect(init!.headers).toEqual({ Authorization: 'Bearer secret' });
		expect(file.name).toBe('audio.wav');
		expect(file.type).toBe('audio/wav');
		expect(new Uint8Array(await file.arrayBuffer())).toEqual(new Uint8Array([1, 2, 3]));
		expect(form.get('model')).toBe('org/model:Q8_0');
		expect(form.get('language')).toBe('en');
		expect(form.get('response_format')).toBe('json');
		expect(form.has('prompt')).toBe(false);
	});

	it('returns the words alone when the model wraps them in a format of its own', async () => {
		const answers = [
			'language French<asr_text>Bonjour.',
			'[{"Start":0.0,"End":3.1,"Speaker":0,"Content":"Bonjour."}]'
		];

		for (const answer of answers) {
			const fetchMock = vi.fn(async () => new Response(JSON.stringify({ text: answer })));

			await expect(
				transcribeClip(contextWith(fetchMock as typeof fetch), new Blob([]))
			).resolves.toBe('Bonjour.');
		}
	});

	it('leaves the model out for a single-model server', async () => {
		const fetchMock = vi.fn(
			async (_url: RequestInfo | URL, _init?: RequestInit) =>
				new Response(JSON.stringify({ text: 'ok' }))
		);

		await transcribeClip(contextWith(fetchMock as typeof fetch), new Blob([]));

		expect((fetchMock.mock.calls[0][1]!.body as FormData).has('model')).toBe(false);
	});

	it("surfaces the server's error message and status", async () => {
		const fetchMock = vi.fn(
			async () =>
				new Response(
					JSON.stringify({
						error: {
							code: 501,
							message: 'The current model does not support audio input.',
							type: 'not_supported_error'
						}
					}),
					{ status: 501 }
				)
		);
		const error = await transcribeClip(contextWith(fetchMock as typeof fetch), new Blob([])).catch(
			(e) => e
		);

		expect(error).toBeInstanceOf(VoiceTranscriptionError);
		expect(error.message).toBe('The current model does not support audio input.');
		expect(error.status).toBe(501);
	});

	it('falls back to the status for bodies that say nothing useful', async () => {
		const fetchMock = vi.fn(
			async () =>
				new Response('<html>Bad Gateway</html>', { status: 502, statusText: 'Bad Gateway' })
		);

		await expect(
			transcribeClip(contextWith(fetchMock as typeof fetch), new Blob([]))
		).rejects.toThrow('Transcription failed (502 Bad Gateway)');
	});

	it('rejects a response without text', async () => {
		const fetchMock = vi.fn(async () => new Response(JSON.stringify({ choices: [] })));

		await expect(
			transcribeClip(contextWith(fetchMock as typeof fetch), new Blob([]))
		).rejects.toBeInstanceOf(VoiceTranscriptionError);
	});

	it('passes the abort signal on', async () => {
		const controller = new AbortController();
		const fetchMock = vi.fn(
			async (_url: RequestInfo | URL, _init?: RequestInit) =>
				new Response(JSON.stringify({ text: '' }))
		);

		await transcribeClip(contextWith(fetchMock as typeof fetch), new Blob([]), {
			signal: controller.signal
		});

		expect(fetchMock.mock.calls[0][1]!.signal).toBe(controller.signal);
	});
});
