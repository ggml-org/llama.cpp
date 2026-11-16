import {
	VOICE_API_PATHS,
	VOICE_REALTIME_EVENTS,
	VOICE_REALTIME_QUERY,
	VOICE_REALTIME_SESSION_TRANSCRIPTION,
	VOICE_REALTIME_TIMEOUTS,
	VOICE_REALTIME_UNSUPPORTED_PATTERN
} from './constants';
import { RealtimeError, RealtimeUnsupportedError } from './errors';
import { pcm16ToBase64 } from './pcm';
import type { RealtimeTranscriberOptions, VoiceServerContext } from './types';

type RealtimeState = 'idle' | 'connecting' | 'open' | 'finishing' | 'closed';

interface Pending<T> {
	resolve: (value: T) => void;
	reject: (error: Error) => void;
}

/** A server event, only the fields this client reads. */
interface RealtimeServerEvent {
	type?: string;
	delta?: unknown;
	transcript?: unknown;
	error?: { message?: unknown };
	session?: { type?: unknown; audio?: { input?: { format?: { rate?: unknown } } } };
}

/**
 * URL of the realtime socket for a server: the same target as its HTTP routes, with
 * ws(s) for http(s), `intent=transcription`, and the model for a router.
 * No API key goes in it: llama-server only reads the key from headers.
 */
export function realtimeUrl(
	context: VoiceServerContext,
	options: Pick<RealtimeTranscriberOptions, 'autoload' | 'model'> = {}
): string {
	const url = new URL(
		context.resolveUrl(VOICE_API_PATHS.REALTIME),
		context.baseHref ?? globalThis.location?.href
	);

	url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
	url.searchParams.set(VOICE_REALTIME_QUERY.INTENT, VOICE_REALTIME_QUERY.INTENT_TRANSCRIPTION);

	if (options.model) url.searchParams.set(VOICE_REALTIME_QUERY.MODEL, options.model);

	if (options.autoload !== undefined) {
		url.searchParams.set(VOICE_REALTIME_QUERY.AUTOLOAD, String(options.autoload));
	}

	return url.toString();
}

/**
 * Live transcription over llama-server's `/v1/realtime` socket. Audio given to
 * `append` before the session starts is queued and sent once it does.
 *
 * A model with no streaming audio encoder gets an `error` event and the socket
 * closes: `connect()` then rejects with RealtimeUnsupportedError and the caller
 * transcribes the clip over HTTP instead. A model that holds realtime conversations
 * is switched to a transcription session, which transcribes each committed piece of
 * audio: commitSegment() at the speaker's pauses gives its text while they go on, and
 * the final commit the rest. One piece is transcribed at a time.
 */
export class RealtimeTranscriber {
	/** True when the session streams text while the audio comes in */
	live = false;
	/** Input rate of the model, from session.created */
	modelSampleRate: number | null = null;

	// audio sent since the last commit: an empty commit is an error on the server
	private audioSinceCommit = false;
	private completedText = '';
	private connecting: Promise<void> | null = null;
	private readonly context: VoiceServerContext;
	private finalSent = false;
	private idleWaiters: (() => void)[] = [];

	// commits whose transcript has not completed yet, and who waits for them
	private inFlight = 0;
	private itemText = '';
	private readonly options: RealtimeTranscriberOptions;
	private pendingConnect: Pending<void> | null = null;
	private pendingFinish: Pending<string> | null = null;
	private queue: string[] = [];
	private sentAudio = false;
	private socket: WebSocket | null = null;
	private state: RealtimeState = 'idle';
	private timer: ReturnType<typeof setTimeout> | null = null;

	/**
	 * Whether a piece can be committed now: a session that transcribes per commit, with
	 * audio since the last commit and no piece still being transcribed.
	 */
	get canCommitSegment(): boolean {
		return this.state === 'open' && !this.live && this.audioSinceCommit && this.inFlight === 0;
	}

	/** True between session.created and finish() or close(). */
	get ready(): boolean {
		return this.state === 'open';
	}

	/** The whole transcript so far, live text of the current item included. */
	get text(): string {
		return this.completedText + this.itemText;
	}

	/** Streams a PCM16 mono chunk at `options.sampleRate`. */
	append(chunk: Int16Array): void {
		if (chunk.length === 0 || this.state === 'finishing' || this.state === 'closed') return;

		const message = JSON.stringify({
			audio: pcm16ToBase64(chunk),
			type: VOICE_REALTIME_EVENTS.APPEND
		});

		this.sentAudio = true;
		this.audioSinceCommit = true;

		if (this.state === 'open') {
			this.socket?.send(message);
		} else {
			this.queue.push(message);
		}
	}

	/** Closes the socket; pending connect or finish calls reject. */
	close(): void {
		if (this.state === 'closed') return;

		this.fail(new RealtimeError('The realtime session was closed.'));
	}

	/** Commits the audio so far as a piece to transcribe now; see canCommitSegment. */
	commitSegment(): void {
		if (!this.canCommitSegment) return;

		this.inFlight++;
		this.audioSinceCommit = false;
		this.send({ type: VOICE_REALTIME_EVENTS.COMMIT });
	}

	/**
	 * Opens the socket and resolves once the session is created.
	 * @throws RealtimeUnsupportedError when the model cannot stream or the socket does not open
	 */
	connect(): Promise<void> {
		if (this.connecting) return this.connecting;

		this.connecting = new Promise<void>((resolve, reject) => {
			this.pendingConnect = { reject, resolve };
		});

		this.state = 'connecting';

		let socket: WebSocket;

		try {
			socket = new WebSocket(realtimeUrl(this.context, this.options));
		} catch (error) {
			this.fail(
				new RealtimeUnsupportedError('The realtime socket could not be opened.', 'connection', {
					cause: error
				})
			);

			return this.connecting;
		}

		this.socket = socket;
		socket.onmessage = (event: MessageEvent) => this.handleMessage(event.data);
		socket.onclose = (event: CloseEvent) => this.handleClose(event);
		this.startTimer(
			this.options.connectTimeoutMs ?? VOICE_REALTIME_TIMEOUTS.CONNECT_MS,
			'The realtime session did not start in time.'
		);

		return this.connecting;
	}

	constructor(context: VoiceServerContext, options: RealtimeTranscriberOptions) {
		this.context = context;
		this.options = options;
	}

	/**
	 * Ends the input (`input_audio_buffer.commit` with `final: true`) and resolves with
	 * the whole transcript once the last item completes. With `commitRest: false`, a
	 * session that transcribes per commit leaves out the audio after the last piece,
	 * e.g. when it is silence, which such a model may hear a word in.
	 * @throws RealtimeError when the session fails, closes or times out first
	 */
	async finish(options: { commitRest?: boolean } = {}): Promise<string> {
		if (this.state === 'connecting') await this.connecting;

		if (this.state !== 'open') throw new RealtimeError('The realtime session is not open.');

		if (!this.sentAudio) {
			this.close();

			return '';
		}

		this.state = 'finishing';

		const done = new Promise<string>((resolve, reject) => {
			this.pendingFinish = { reject, resolve };
		});

		this.startTimer(
			this.options.finishTimeoutMs ?? VOICE_REALTIME_TIMEOUTS.FINISH_MS,
			'The transcript did not complete in time.'
		);

		// a piece still being transcribed finishes first: the server follows one at a time
		while (this.inFlight > 0 && this.state === 'finishing') {
			await new Promise<void>((resolve) => this.idleWaiters.push(resolve));
		}

		if (this.state !== 'finishing') return done;

		const commitRest = this.live || options.commitRest !== false || this.completedText === '';

		if (this.audioSinceCommit && commitRest) {
			this.finalSent = true;
			this.inFlight++;
			this.audioSinceCommit = false;
			this.send({ final: true, type: VOICE_REALTIME_EVENTS.COMMIT });
		} else {
			// everything was committed in pieces already, or what is left holds no speech
			this.resolveFinish();
		}

		return done;
	}

	private clearTimer(): void {
		if (this.timer !== null) {
			clearTimeout(this.timer);
			this.timer = null;
		}
	}

	/** Rejects whatever is pending with `error` and releases the socket. */
	private fail(error: Error): void {
		this.state = 'closed';
		this.clearTimer();

		for (const wake of this.idleWaiters.splice(0)) wake();

		this.queue = [];
		this.pendingConnect?.reject(error);
		this.pendingConnect = null;
		this.pendingFinish?.reject(error);
		this.pendingFinish = null;

		const socket = this.socket;

		this.socket = null;

		if (socket) {
			socket.onmessage = null;
			socket.onclose = null;
			socket.close();
		}
	}

	private handleClose(event: CloseEvent): void {
		if (this.state === 'closed') return;

		// a refused upgrade (e.g. 401 from a server with an API key) closes before any session
		const error =
			this.state === 'connecting'
				? new RealtimeUnsupportedError(
						`The realtime socket closed before the session started (code ${event.code}).`,
						'connection'
					)
				: new RealtimeError('The realtime socket closed before the transcript completed.');

		if (this.state === 'open') this.options.onError?.(error);

		this.fail(error);
	}

	private handleCompleted(transcript: string): void {
		this.completedText = joinPieces(this.completedText, transcript);
		this.itemText = '';
		this.inFlight = Math.max(0, this.inFlight - 1);
		this.options.onCompleted?.(this.text);

		if (this.inFlight === 0) {
			for (const wake of this.idleWaiters.splice(0)) wake();
		}

		// a live session completes once, on its final commit; a per-commit one once the last piece is in
		if (this.state === 'finishing' && (this.live || (this.finalSent && this.inFlight === 0))) {
			this.resolveFinish();
		}
	}

	private handleError(message: string): void {
		if (this.state === 'connecting') {
			this.fail(
				VOICE_REALTIME_UNSUPPORTED_PATTERN.test(message)
					? new RealtimeUnsupportedError(message, 'model')
					: new RealtimeError(message)
			);

			return;
		}

		const error = new RealtimeError(message);

		this.options.onError?.(error);

		if (this.state === 'finishing') this.fail(error);
	}

	private handleMessage(data: unknown): void {
		if (typeof data !== 'string') return;

		let event: RealtimeServerEvent;

		try {
			event = JSON.parse(data);
		} catch {
			return;
		}

		switch (event.type) {
			case VOICE_REALTIME_EVENTS.SESSION_CREATED:
				this.handleSessionCreated(event);

				break;
			case VOICE_REALTIME_EVENTS.DELTA:
				if (typeof event.delta === 'string' && event.delta) {
					this.itemText += event.delta;
					this.options.onDelta?.(event.delta, this.text);
				}

				break;
			case VOICE_REALTIME_EVENTS.COMPLETED:
				this.handleCompleted(
					typeof event.transcript === 'string' ? event.transcript : this.itemText
				);

				break;
			case VOICE_REALTIME_EVENTS.ERROR:
				this.handleError(
					typeof event.error?.message === 'string'
						? event.error.message
						: 'Realtime transcription failed.'
				);

				break;
		}
	}

	private handleSessionCreated(event: RealtimeServerEvent): void {
		if (this.state !== 'connecting') return;

		const rate = event.session?.audio?.input?.format?.rate;

		this.modelSampleRate = typeof rate === 'number' ? rate : null;
		this.live = event.session?.type === VOICE_REALTIME_SESSION_TRANSCRIPTION;
		this.clearTimer();

		// the rate of our chunks, and a transcription session for a model that would hold a conversation
		this.send({
			session: {
				audio: {
					input: {
						format: { rate: this.options.sampleRate, type: 'audio/pcm' },
						...(this.options.language ? { transcription: { language: this.options.language } } : {})
					}
				},
				type: VOICE_REALTIME_SESSION_TRANSCRIPTION
			},
			type: VOICE_REALTIME_EVENTS.SESSION_UPDATE
		});

		this.state = 'open';

		for (const message of this.queue) this.socket?.send(message);

		this.queue = [];
		this.pendingConnect?.resolve();
		this.pendingConnect = null;
	}

	private resolveFinish(): void {
		if (!this.pendingFinish) return;

		const text = this.text.trim();

		this.pendingFinish.resolve(text);
		this.pendingFinish = null;
		this.close();
	}

	private send(event: object): void {
		this.socket?.send(JSON.stringify(event));
	}

	private startTimer(ms: number, message: string): void {
		this.clearTimer();
		this.timer = setTimeout(() => this.fail(new RealtimeError(message)), ms);
	}
}

/** Two transcribed pieces as one text, a space between them where neither has one. */
function joinPieces(before: string, piece: string): string {
	if (!before) return piece;

	if (!piece) return before;

	return /\s$/.test(before) || /^\s/.test(piece) ? before + piece : `${before} ${piece}`;
}
