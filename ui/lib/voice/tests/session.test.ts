// Guards the voice session: audio goes to the realtime socket while recording and the
// final transcript comes from it; when the model cannot stream, the session falls back
// to one HTTP transcription of the clip without telling the user anything went wrong.

import { type VoiceCapture } from '../src/capture.js';
import { startVoiceSession } from '../src/session.js';
import { type VoiceClip, type VoicePcmListener, type VoiceServerContext } from '../src/types.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

class FakeWebSocket {
	closed = false;

	static instances: FakeWebSocket[] = [];
	onclose: ((event: { code: number }) => void) | null = null;
	onmessage: ((event: { data: string }) => void) | null = null;
	sent: Record<string, unknown>[] = [];

	close() {
		this.closed = true;
	}

	constructor(public url: string) {
		FakeWebSocket.instances.push(this);
	}

	emit(event: Record<string, unknown>) {
		this.onmessage?.({ data: JSON.stringify(event) });
	}

	send(data: string) {
		this.sent.push(JSON.parse(data));
	}
}

const clip: VoiceClip = {
	blob: new Blob([new Uint8Array(44)], { type: 'audio/wav' }),
	durationMs: 1000,
	peaks: [0, 1],
	sampleRate: 16_000
};

function fakeCapture() {
	const pcm = new Set<VoicePcmListener>();
	const levels = new Set<(levels: number[]) => void>();
	const capture = {
		cancel: vi.fn(),
		levels: [0, 0],
		levelsIntervalMs: 50,
		onLevels: vi.fn((listener: (levels: number[]) => void) => {
			levels.add(listener);

			return () => levels.delete(listener);
		}),
		onPcm: (listener: VoicePcmListener) => {
			pcm.add(listener);

			return () => pcm.delete(listener);
		},
		sampleRate: 16_000,
		start: vi.fn(async () => {}),
		stop: vi.fn(async () => {
			// the last chunk reaches the listeners before stop resolves, as in VoiceCapture
			for (const listener of pcm) listener(Int16Array.from([9]));

			return clip;
		})
	};

	return {
		capture: capture as unknown as VoiceCapture,
		emitPcm: (chunk: Int16Array) => pcm.forEach((listener) => listener(chunk)),
		/** Holds one input level for `ms`, as the capture reports it every 50 ms. */
		holdLevel: (level: number, ms: number) => {
			for (let t = 0; t < ms; t += 50) levels.forEach((listener) => listener([level]));
		},
		raw: capture
	};
}

function serverContext(fetchImpl: typeof fetch): VoiceServerContext {
	return {
		baseHref: 'http://localhost/',
		fetch: fetchImpl,
		resolveUrl: (path) => new URL(path, 'http://localhost/').toString()
	};
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

beforeEach(() => {
	vi.stubGlobal('WebSocket', FakeWebSocket);
});

afterEach(() => {
	FakeWebSocket.instances = [];
	vi.unstubAllGlobals();
});

describe('startVoiceSession', () => {
	it('records only when not asked to transcribe', async () => {
		const { capture } = fakeCapture();
		const session = await startVoiceSession({ capture });

		expect(FakeWebSocket.instances).toHaveLength(0);
		await expect(session.stop()).resolves.toEqual({ clip, liveText: null, transcript: null });
	});

	it('streams to the realtime socket and takes its transcript', async () => {
		const fetchMock = vi.fn();
		const onText = vi.fn();
		const onRealtime = vi.fn();
		const { capture, emitPcm } = fakeCapture();
		const session = await startVoiceSession({
			capture,
			onRealtime,
			onText,
			transcribe: { context: serverContext(fetchMock as typeof fetch), model: 'asr' }
		});
		const ws = FakeWebSocket.instances[0];

		expect(ws.url).toBe('ws://localhost/v1/realtime?intent=transcription&model=asr');

		emitPcm(Int16Array.from([1]));
		ws.emit({ session: { type: 'transcription' }, type: 'session.created' });
		await flush();

		expect(onRealtime).toHaveBeenCalledWith(true);

		ws.emit({ delta: 'Hi', type: 'conversation.item.input_audio_transcription.delta' });
		expect(onText).toHaveBeenLastCalledWith('Hi');

		const result = await session.stop();

		expect(result.clip).toBe(clip);
		expect(result.liveText).toBe('Hi');
		// the queued chunk, the last chunk from stop(), then the final commit
		expect(ws.sent.map((event) => event.type)).toEqual([
			'session.update',
			'input_audio_buffer.append',
			'input_audio_buffer.append',
			'input_audio_buffer.commit'
		]);

		ws.emit({ transcript: 'Hi.', type: 'conversation.item.input_audio_transcription.completed' });

		await expect(result.transcript).resolves.toBe('Hi.');
		expect(fetchMock).not.toHaveBeenCalled();
	});

	describe('a recording transcribed in pieces', () => {
		const COMPLETED = 'conversation.item.input_audio_transcription.completed';

		/** Speaks two pieces with a pause between them and stops. */
		async function speakTwoPieces(fetchMock: ReturnType<typeof vi.fn>) {
			const onText = vi.fn();
			const { capture, emitPcm, holdLevel } = fakeCapture();
			const session = await startVoiceSession({
				capture,
				onText,
				transcribe: { context: serverContext(fetchMock as unknown as typeof fetch) }
			});
			const ws = FakeWebSocket.instances[0];

			// a model that holds conversations transcribes per commit
			ws.emit({ session: { type: 'realtime' }, type: 'session.created' });
			await flush();

			emitPcm(Int16Array.from([1]));
			holdLevel(0.05, 200);
			holdLevel(0.6, 500);
			holdLevel(0.05, 400);

			expect(ws.sent.filter((event) => event.type === 'input_audio_buffer.commit')).toHaveLength(1);

			ws.emit({ transcript: 'I went.', type: COMPLETED });
			expect(onText).toHaveBeenLastCalledWith('I went.');

			emitPcm(Int16Array.from([2]));
			holdLevel(0.6, 500);

			const result = await session.stop();

			ws.emit({ transcript: 'To the store.', type: COMPLETED });

			return result;
		}

		it('is transcribed once more as a whole, punctuated by meaning', async () => {
			const fetchMock = vi.fn(
				async () => new Response(JSON.stringify({ text: 'I went to the store.' }))
			);
			const result = await speakTwoPieces(fetchMock);

			expect(result.liveText).toBe('I went.');
			await expect(result.transcript).resolves.toBe('I went to the store.');
			expect(fetchMock).toHaveBeenCalledOnce();
		});

		it('keeps its pieces when the whole pass fails', async () => {
			const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
			const fetchMock = vi.fn(async () => new Response('busy', { status: 503 }));
			const result = await speakTwoPieces(fetchMock);

			await expect(result.transcript).resolves.toBe('I went. To the store.');
			warn.mockRestore();
		});
	});

	it('falls back to HTTP when the model cannot stream', async () => {
		const fetchMock = vi.fn(async () => new Response(JSON.stringify({ text: 'From HTTP.' })));
		const onRealtime = vi.fn();
		const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
		const { capture } = fakeCapture();
		const session = await startVoiceSession({
			capture,
			onRealtime,
			transcribe: { context: serverContext(fetchMock as typeof fetch), model: 'omni' }
		});

		FakeWebSocket.instances[0].emit({
			error: { message: 'The current model does not support realtime transcription.' },
			type: 'error'
		});
		await flush();

		expect(onRealtime).toHaveBeenCalledWith(false);
		// an expected fallback, not worth a warning
		expect(warn).not.toHaveBeenCalled();

		const result = await session.stop();

		expect(result.liveText).toBeNull();
		await expect(result.transcript).resolves.toBe('From HTTP.');
		expect(fetchMock).toHaveBeenCalledOnce();

		const form = (fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1].body as FormData;

		expect(form.get('model')).toBe('omni');
		warn.mockRestore();
	});

	describe('a model that cannot stream (Qwen3-ASR)', () => {
		/** Answers a piece with "Piece N." and the whole clip with `whole`, or fails it. */
		function server(whole: string | null) {
			let pieces = 0;

			return vi.fn(async (_url: string, init: RequestInit) => {
				const file = (init.body as FormData).get('file') as File;

				if (file.name === 'piece.wav') {
					return new Response(JSON.stringify({ text: `Piece ${++pieces}.` }));
				}

				return whole === null
					? new Response('busy', { status: 503 })
					: new Response(JSON.stringify({ text: whole }));
			});
		}

		/** Speaks two pieces with a pause after each, over HTTP once the socket says no. */
		async function speakOverHttp(fetchMock: ReturnType<typeof server>) {
			const onText = vi.fn();
			const { capture, emitPcm, holdLevel } = fakeCapture();
			const session = await startVoiceSession({
				capture,
				onText,
				transcribe: { context: serverContext(fetchMock as unknown as typeof fetch), model: 'asr' }
			});

			// said before the socket answered: kept for the first piece
			emitPcm(Int16Array.from([1, 2]));
			FakeWebSocket.instances[0].emit({
				error: { message: 'The current model does not support realtime transcription.' },
				type: 'error'
			});
			await flush();

			holdLevel(0.05, 200);
			holdLevel(0.6, 500);
			holdLevel(0.05, 400);
			await flush();
			expect(onText).toHaveBeenLastCalledWith('Piece 1.');

			emitPcm(Int16Array.from([3]));
			holdLevel(0.6, 500);
			holdLevel(0.05, 400);
			await flush();
			expect(onText).toHaveBeenLastCalledWith('Piece 1. Piece 2.');

			return session.stop();
		}

		it('gives live text piece by piece over HTTP, then the whole clip', async () => {
			const fetchMock = server('Whole.');
			const result = await speakOverHttp(fetchMock);

			expect(result.liveText).toBe('Piece 1. Piece 2.');
			await expect(result.transcript).resolves.toBe('Whole.');
			// two pieces and the whole clip
			expect(fetchMock).toHaveBeenCalledTimes(3);

			const first = (fetchMock.mock.calls[0][1].body as FormData).get('file') as File;

			// the audio from before the socket answered went with the first piece
			expect(first.size).toBeGreaterThan(44);
		});

		it('keeps the pieces when the whole clip fails', async () => {
			const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
			const result = await speakOverHttp(server(null));

			await expect(result.transcript).resolves.toBe('Piece 1. Piece 2.');
			warn.mockRestore();
		});
	});

	it('uses HTTP when realtime is still connecting at the end', async () => {
		const fetchMock = vi.fn(async () => new Response(JSON.stringify({ text: 'Late.' })));
		const { capture } = fakeCapture();
		const session = await startVoiceSession({
			capture,
			transcribe: { context: serverContext(fetchMock as typeof fetch) }
		});
		const result = await session.stop();

		expect(FakeWebSocket.instances[0].closed).toBe(true);
		await expect(result.transcript).resolves.toBe('Late.');
	});

	it('skips the socket when realtime is off', async () => {
		const fetchMock = vi.fn(async () => new Response(JSON.stringify({ text: 'ok' })));
		const { capture } = fakeCapture();
		const session = await startVoiceSession({
			capture,
			realtime: false,
			transcribe: { context: serverContext(fetchMock as typeof fetch) }
		});

		expect(FakeWebSocket.instances).toHaveLength(0);
		await expect((await session.stop()).transcript).resolves.toBe('ok');
	});

	it('cancel drops the capture and the socket', async () => {
		const { capture, raw } = fakeCapture();
		const session = await startVoiceSession({
			capture,
			transcribe: { context: serverContext(vi.fn() as unknown as typeof fetch) }
		});

		session.cancel();

		expect(raw.cancel).toHaveBeenCalledOnce();
		expect(FakeWebSocket.instances[0].closed).toBe(true);
	});
});
