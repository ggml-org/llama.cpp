// Guards the realtime transcription client against llama-server's /v1/realtime
// protocol (tools/server/README.md): the URL, the session flow, deltas and completed
// events, the final commit, the "does not support" error, refused sockets and timeouts.

import { RealtimeError, RealtimeUnsupportedError } from '../src/errors.js';
import { base64ToBytes } from '../src/pcm.js';
import { RealtimeTranscriber, realtimeUrl } from '../src/realtime.js';
import { type VoiceServerContext } from '../src/types.js';
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

	/** The server (or the browser, on a refused upgrade) closes the socket. */
	drop(code = 1006) {
		this.onclose?.({ code });
	}

	/** An event from the server. */
	emit(event: Record<string, unknown>) {
		this.onmessage?.({ data: JSON.stringify(event) });
	}

	send(data: string) {
		this.sent.push(JSON.parse(data));
	}
}

const context: VoiceServerContext = {
	baseHref: 'http://localhost:8080/',
	resolveUrl: (path) => new URL(path, 'http://localhost:8080/').toString()
};
const sessionCreated = (type = 'transcription') => ({
	session: { audio: { input: { format: { rate: 24_000, type: 'audio/pcm' } } }, type },
	type: 'session.created'
});

function socket(): FakeWebSocket {
	return FakeWebSocket.instances.at(-1)!;
}

beforeEach(() => {
	vi.stubGlobal('WebSocket', FakeWebSocket);
});

afterEach(() => {
	FakeWebSocket.instances = [];
	vi.unstubAllGlobals();
	vi.useRealTimers();
});

describe('realtimeUrl', () => {
	it('turns the HTTP target into a websocket URL with the intent and model', () => {
		expect(realtimeUrl(context, { model: 'org/model:Q4' })).toBe(
			'ws://localhost:8080/v1/realtime?intent=transcription&model=org%2Fmodel%3AQ4'
		);
	});

	it('uses wss for https and resolves relative targets against the page', () => {
		const relative: VoiceServerContext = {
			baseHref: 'https://example.com/llama/',
			resolveUrl: (path) => path
		};

		expect(realtimeUrl(relative, { autoload: false })).toBe(
			'wss://example.com/llama/v1/realtime?intent=transcription&autoload=false'
		);
	});
});

describe('RealtimeTranscriber', () => {
	it('starts the session, sends the capture rate, and flushes audio queued while connecting', async () => {
		const transcriber = new RealtimeTranscriber(context, { language: 'en', sampleRate: 16_000 });
		const connected = transcriber.connect();

		transcriber.append(Int16Array.from([1, 2]));
		expect(socket().sent).toEqual([]);

		socket().emit(sessionCreated());
		await connected;

		expect(transcriber.ready).toBe(true);
		expect(transcriber.live).toBe(true);
		expect(transcriber.modelSampleRate).toBe(24_000);
		expect(socket().sent[0]).toEqual({
			session: {
				audio: {
					input: {
						format: { rate: 16_000, type: 'audio/pcm' },
						transcription: { language: 'en' }
					}
				},
				type: 'transcription'
			},
			type: 'session.update'
		});
		expect(socket().sent[1].type).toBe('input_audio_buffer.append');
		expect(Array.from(base64ToBytes(socket().sent[1].audio as string))).toEqual([1, 0, 2, 0]);

		transcriber.append(Int16Array.from([3]));
		expect(socket().sent).toHaveLength(3);
	});

	it('streams deltas, ends with a final commit and resolves with the completed transcript', async () => {
		const onDelta = vi.fn();
		const onCompleted = vi.fn();
		const transcriber = new RealtimeTranscriber(context, {
			onCompleted,
			onDelta,
			sampleRate: 16_000
		});
		const connected = transcriber.connect();

		socket().emit(sessionCreated());
		await connected;
		transcriber.append(Int16Array.from([1]));

		socket().emit({ delta: 'Hello', type: 'conversation.item.input_audio_transcription.delta' });
		socket().emit({ delta: ' world', type: 'conversation.item.input_audio_transcription.delta' });

		expect(onDelta).toHaveBeenLastCalledWith(' world', 'Hello world');
		expect(transcriber.text).toBe('Hello world');

		const finished = transcriber.finish();

		expect(socket().sent.at(-1)).toEqual({ final: true, type: 'input_audio_buffer.commit' });

		socket().emit({
			transcript: 'Hello world.',
			type: 'conversation.item.input_audio_transcription.completed'
		});

		await expect(finished).resolves.toBe('Hello world.');
		expect(onCompleted).toHaveBeenCalledWith('Hello world.');
		expect(socket().closed).toBe(true);
		expect(transcriber.ready).toBe(false);
	});

	it('joins the items completed before the end', async () => {
		const transcriber = new RealtimeTranscriber(context, { sampleRate: 16_000 });
		const connected = transcriber.connect();

		socket().emit(sessionCreated());
		await connected;
		transcriber.append(Int16Array.from([1]));
		socket().emit({
			transcript: 'First line.\n',
			type: 'conversation.item.input_audio_transcription.completed'
		});
		socket().emit({ delta: 'Second', type: 'conversation.item.input_audio_transcription.delta' });

		expect(transcriber.text).toBe('First line.\nSecond');

		const finished = transcriber.finish();

		socket().emit({
			transcript: 'Second line.',
			type: 'conversation.item.input_audio_transcription.completed'
		});

		await expect(finished).resolves.toBe('First line.\nSecond line.');
	});

	it('shows the words alone while a model streams them in a format of its own', async () => {
		const onDelta = vi.fn();
		const transcriber = new RealtimeTranscriber(context, { onDelta, sampleRate: 16_000 });
		const connected = transcriber.connect();
		const delta = (text: string) =>
			socket().emit({ delta: text, type: 'conversation.item.input_audio_transcription.delta' });
		const completed = (transcript: string) =>
			socket().emit({ transcript, type: 'conversation.item.input_audio_transcription.completed' });

		socket().emit(sessionCreated('realtime'));
		await connected;
		transcriber.append(Int16Array.from([1]));
		transcriber.commitSegment();

		delta('[{"Start":0.0,"End":1.2,"Spe');
		expect(transcriber.text).toBe('');
		delta('aker":0,"Content":"Bonjour');
		expect(onDelta).toHaveBeenLastCalledWith('aker":0,"Content":"Bonjour', 'Bonjour');
		completed('[{"Start":0.0,"End":1.2,"Speaker":0,"Content":"Bonjour."}]');
		expect(transcriber.text).toBe('Bonjour.');

		// the next piece follows the first, a space between them
		transcriber.append(Int16Array.from([2]));
		transcriber.commitSegment();
		delta('[{"Start":0.0,"End":0.9,"Speaker":0,"Content":"Ça va');
		expect(transcriber.text).toBe('Bonjour. Ça va');

		const finished = transcriber.finish({ commitRest: false });

		completed('language French<asr_text>Ça va ?');
		await expect(finished).resolves.toBe('Bonjour. Ça va ?');
	});

	it('transcribes a conversation model piece by piece, one at a time', async () => {
		const transcriber = new RealtimeTranscriber(context, { sampleRate: 16_000 });
		const connected = transcriber.connect();
		const commits = () =>
			socket().sent.filter((event) => event.type === 'input_audio_buffer.commit');
		const completed = (transcript: string) =>
			socket().emit({ transcript, type: 'conversation.item.input_audio_transcription.completed' });

		socket().emit(sessionCreated('realtime'));
		await connected;

		// nothing to commit before any audio
		expect(transcriber.canCommitSegment).toBe(false);
		transcriber.append(Int16Array.from([1]));
		transcriber.commitSegment();
		expect(commits()).toEqual([{ type: 'input_audio_buffer.commit' }]);

		// the next piece waits for the text of this one
		transcriber.append(Int16Array.from([2]));
		expect(transcriber.canCommitSegment).toBe(false);
		completed('Hello there.');
		expect(transcriber.text).toBe('Hello there.');
		expect(transcriber.canCommitSegment).toBe(true);

		transcriber.commitSegment();
		transcriber.append(Int16Array.from([3]));

		// the end waits for the piece in flight, then commits the rest
		const finished = transcriber.finish();

		await Promise.resolve();
		expect(commits()).toHaveLength(2);
		completed('How are you?');
		await vi.waitFor(() => expect(commits()).toHaveLength(3));
		expect(commits()[2]).toEqual({ final: true, type: 'input_audio_buffer.commit' });
		completed('Fine.');

		await expect(finished).resolves.toBe('Hello there. How are you? Fine.');
	});

	it('goes on with the next piece after one the server failed', async () => {
		const onError = vi.fn();
		const transcriber = new RealtimeTranscriber(context, { onError, sampleRate: 16_000 });
		const connected = transcriber.connect();

		socket().emit(sessionCreated('realtime'));
		await connected;
		transcriber.append(Int16Array.from([1]));
		transcriber.commitSegment();
		socket().emit({ error: { message: 'decode failed' }, type: 'error' });

		expect(onError).toHaveBeenCalledOnce();
		transcriber.append(Int16Array.from([2]));
		expect(transcriber.canCommitSegment).toBe(true);
	});

	it('ends without a final commit when every piece was committed already', async () => {
		const transcriber = new RealtimeTranscriber(context, { sampleRate: 16_000 });
		const connected = transcriber.connect();

		socket().emit(sessionCreated('realtime'));
		await connected;
		transcriber.append(Int16Array.from([1]));
		transcriber.commitSegment();
		socket().emit({
			transcript: 'Only this.',
			type: 'conversation.item.input_audio_transcription.completed'
		});

		await expect(transcriber.finish()).resolves.toBe('Only this.');
		expect(
			socket().sent.filter((event) => event.type === 'input_audio_buffer.commit')
		).toHaveLength(1);
	});

	it('leaves out a silent rest after the last piece, but never the only one', async () => {
		const commits = () =>
			socket().sent.filter((event) => event.type === 'input_audio_buffer.commit');
		const transcriber = new RealtimeTranscriber(context, { sampleRate: 16_000 });
		const connected = transcriber.connect();

		socket().emit(sessionCreated('realtime'));
		await connected;
		transcriber.append(Int16Array.from([1]));
		transcriber.commitSegment();
		socket().emit({
			transcript: 'All said.',
			type: 'conversation.item.input_audio_transcription.completed'
		});
		// silence after the last piece: a model may hear "yeah" in it
		transcriber.append(Int16Array.from([0]));

		await expect(transcriber.finish({ commitRest: false })).resolves.toBe('All said.');
		expect(commits()).toHaveLength(1);

		// with nothing transcribed yet, the rest is all there is: it is committed
		const second = new RealtimeTranscriber(context, { sampleRate: 16_000 });
		const secondConnected = second.connect();

		socket().emit(sessionCreated('realtime'));
		await secondConnected;
		second.append(Int16Array.from([1]));

		const finished = second.finish({ commitRest: false });

		await vi.waitFor(() => expect(commits()).toHaveLength(1));
		socket().emit({
			transcript: 'Short.',
			type: 'conversation.item.input_audio_transcription.completed'
		});
		await expect(finished).resolves.toBe('Short.');
	});

	it('marks a conversation session switched to transcription as not live', async () => {
		const transcriber = new RealtimeTranscriber(context, { sampleRate: 16_000 });
		const connected = transcriber.connect();

		socket().emit(sessionCreated('realtime'));
		await connected;

		expect(transcriber.live).toBe(false);
		expect((socket().sent[0].session as { type: string }).type).toBe('transcription');
	});

	it('rejects with RealtimeUnsupportedError when the model cannot stream', async () => {
		const transcriber = new RealtimeTranscriber(context, { sampleRate: 16_000 });
		const connected = transcriber.connect();

		socket().emit({
			error: {
				code: null,
				message: 'The current model does not support realtime transcription.',
				type: 'invalid_request_error'
			},
			type: 'error'
		});
		socket().drop(1000);

		const error = await connected.catch((e) => e);

		expect(error).toBeInstanceOf(RealtimeUnsupportedError);
		expect(error.reason).toBe('model');
		expect(socket().closed).toBe(true);
	});

	it('rejects with a connection RealtimeUnsupportedError when the upgrade is refused', async () => {
		const transcriber = new RealtimeTranscriber(context, { sampleRate: 16_000 });
		const connected = transcriber.connect();

		// e.g. a 401 from a server with an API key: the browser only reports a close
		socket().drop(1006);

		const error = await connected.catch((e) => e);

		expect(error).toBeInstanceOf(RealtimeUnsupportedError);
		expect(error.reason).toBe('connection');
	});

	it('passes other errors before the session through as RealtimeError', async () => {
		const transcriber = new RealtimeTranscriber(context, { model: 'x', sampleRate: 16_000 });
		const connected = transcriber.connect();

		socket().emit({
			error: { message: 'model name=x is not running', type: 'server_error' },
			type: 'error'
		});

		const error = await connected.catch((e) => e);

		expect(error).toBeInstanceOf(RealtimeError);
		expect(error).not.toBeInstanceOf(RealtimeUnsupportedError);
		expect(error.message).toBe('model name=x is not running');
	});

	it('times out a session that never starts', async () => {
		vi.useFakeTimers();

		const transcriber = new RealtimeTranscriber(context, {
			connectTimeoutMs: 1000,
			sampleRate: 16_000
		});
		const connected = transcriber.connect().catch((e) => e);

		vi.advanceTimersByTime(1000);

		expect(await connected).toBeInstanceOf(RealtimeError);
		expect(socket().closed).toBe(true);
	});

	it('times out a transcript that never completes', async () => {
		vi.useFakeTimers();

		const transcriber = new RealtimeTranscriber(context, {
			finishTimeoutMs: 500,
			sampleRate: 16_000
		});
		const connected = transcriber.connect();

		socket().emit(sessionCreated());
		await connected;
		transcriber.append(Int16Array.from([1]));

		const finished = transcriber.finish().catch((e) => e);

		vi.advanceTimersByTime(500);

		const error = await finished;

		expect(error).toBeInstanceOf(RealtimeError);
		expect(error.message).toMatch(/did not complete/);
	});

	it('rejects finish when the socket closes or errors before the transcript', async () => {
		const onError = vi.fn();
		const transcriber = new RealtimeTranscriber(context, { onError, sampleRate: 16_000 });
		const connected = transcriber.connect();

		socket().emit(sessionCreated());
		await connected;
		transcriber.append(Int16Array.from([1]));

		const finished = transcriber.finish();

		socket().emit({ error: { message: 'server error' }, type: 'error' });

		await expect(finished).rejects.toThrow('server error');
		expect(onError).toHaveBeenCalledOnce();
	});

	it('resolves an empty transcript without a commit when no audio was sent', async () => {
		const transcriber = new RealtimeTranscriber(context, { sampleRate: 16_000 });
		const connected = transcriber.connect();

		socket().emit(sessionCreated());
		await connected;

		await expect(transcriber.finish()).resolves.toBe('');
		expect(socket().sent.map((event) => event.type)).toEqual(['session.update']);
		expect(socket().closed).toBe(true);
	});

	it('close() rejects a pending connect and stops sending', async () => {
		const transcriber = new RealtimeTranscriber(context, { sampleRate: 16_000 });
		const connected = transcriber.connect().catch((e) => e);
		const ws = socket();

		transcriber.close();
		transcriber.append(Int16Array.from([1]));

		expect(await connected).toBeInstanceOf(RealtimeError);
		expect(ws.closed).toBe(true);
		expect(ws.sent).toEqual([]);
	});
});
