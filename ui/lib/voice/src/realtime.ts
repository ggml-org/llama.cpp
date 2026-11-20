import {
	VOICE_API_PATHS,
	VOICE_ERROR_MESSAGE,
	VOICE_LEADING_SPACE_PATTERN,
	VOICE_REALTIME_AUDIO_FORMAT,
	VOICE_REALTIME_EVENTS,
	VOICE_REALTIME_QUERY,
	VOICE_REALTIME_SESSION_TRANSCRIPTION,
	VOICE_REALTIME_TIMEOUTS,
	VOICE_REALTIME_UNSUPPORTED_PATTERN,
	VOICE_SOCKET_PROTOCOL,
	VOICE_TRAILING_SPACE_PATTERN
} from './constants.js';
import { RealtimeState, RealtimeUnsupportedReason } from './enums.js';
import { RealtimeError, RealtimeUnsupportedError } from './errors.js';
import { pcm16ToBase64 } from './pcm.js';
import { cleanTranscript } from './transcript.js';
import type { RealtimeTranscriberOptions, VoiceServerContext } from './types.js';

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
	session?: {
		type?: unknown;
		audio?: { input?: { format?: { rate?: unknown } } };
	};
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

	url.protocol =
		url.protocol === VOICE_SOCKET_PROTOCOL.HTTPS
			? VOICE_SOCKET_PROTOCOL.WSS
			: VOICE_SOCKET_PROTOCOL.WS;
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
	private pieces = 0;
	private queue: string[] = [];
	private sentAudio = false;
	private socket: WebSocket | null = null;
	private state: RealtimeState = RealtimeState.IDLE;
	private timer: ReturnType<typeof setTimeout> | null = null;

	/**
	 * Whether a piece can be committed now: a session that transcribes per commit, with
	 * audio since the last commit and no piece still being transcribed.
	 */
	get canCommitSegment(): boolean {
		return (
			this.state === RealtimeState.OPEN &&
			!this.live &&
			this.audioSinceCommit &&
			this.inFlight === 0
		);
	}

	/** Pieces committed with commitSegment(), the final commit not counted. */
	get piecesCommitted(): number {
		return this.pieces;
	}

	/** True between session.created and finish() or close(). */
	get ready(): boolean {
		return this.state === RealtimeState.OPEN;
	}

	/**
	 * The whole transcript so far, live text of the current item included, without the
	 * format a model wraps its text in (see cleanTranscript).
	 */
	get text(): string {
		const item = cleanTranscript(this.itemText);

		// a model's own format carries no spacing between items; plain text streams with its own
		return item === this.itemText
			? this.completedText + item
			: joinPieces(this.completedText, item);
	}

	/** Streams a PCM16 mono chunk at `options.sampleRate`. */
	append(chunk: Int16Array): void {
		if (
			chunk.length === 0 ||
			this.state === RealtimeState.FINISHING ||
			this.state === RealtimeState.CLOSED
		)
			return;

		const message = JSON.stringify({
			audio: pcm16ToBase64(chunk),
			type: VOICE_REALTIME_EVENTS.APPEND
		});

		this.sentAudio = true;
		this.audioSinceCommit = true;

		if (this.state === RealtimeState.OPEN) {
			this.socket?.send(message);
		} else {
			this.queue.push(message);
		}
	}

	/** Closes the socket; pending connect or finish calls reject. */
	close(): void {
		if (this.state === RealtimeState.CLOSED) return;

		this.fail(new RealtimeError(VOICE_ERROR_MESSAGE.realtimeClosed));
	}

	/** Commits the audio so far as a piece to transcribe now; see canCommitSegment. */
	commitSegment(): void {
		if (!this.canCommitSegment) return;

		this.inFlight++;
		this.pieces++;
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

		this.state = RealtimeState.CONNECTING;

		let socket: WebSocket;

		try {
			socket = new WebSocket(realtimeUrl(this.context, this.options));
		} catch (error) {
			this.fail(
				new RealtimeUnsupportedError(
					VOICE_ERROR_MESSAGE.realtimeSocketFailed,
					RealtimeUnsupportedReason.CONNECTION,
					{ cause: error }
				)
			);

			return this.connecting;
		}

		this.socket = socket;
		socket.onmessage = (event: MessageEvent) => this.handleMessage(event.data);
		socket.onclose = (event: CloseEvent) => this.handleClose(event);
		this.startTimer(
			this.options.connectTimeoutMs ?? VOICE_REALTIME_TIMEOUTS.CONNECT_MS,
			VOICE_ERROR_MESSAGE.realtimeConnectTimeout
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
		if (this.state === RealtimeState.CONNECTING) await this.connecting;

		if (this.state !== RealtimeState.OPEN)
			throw new RealtimeError(VOICE_ERROR_MESSAGE.realtimeNotOpen);

		if (!this.sentAudio) {
			this.close();

			return '';
		}

		this.state = RealtimeState.FINISHING;

		const done = new Promise<string>((resolve, reject) => {
			this.pendingFinish = { reject, resolve };
		});

		this.startTimer(
			this.options.finishTimeoutMs ?? VOICE_REALTIME_TIMEOUTS.FINISH_MS,
			VOICE_ERROR_MESSAGE.realtimeFinishTimeout
		);

		// a piece still being transcribed finishes first: the server follows one at a time
		while (this.inFlight > 0 && this.state === RealtimeState.FINISHING) {
			await new Promise<void>((resolve) => this.idleWaiters.push(resolve));
		}

		if (this.state !== RealtimeState.FINISHING) return done;

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
		this.state = RealtimeState.CLOSED;
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
		if (this.state === RealtimeState.CLOSED) return;

		// a refused upgrade (e.g. 401 from a server with an API key) closes before any session
		const error =
			this.state === RealtimeState.CONNECTING
				? new RealtimeUnsupportedError(
						VOICE_ERROR_MESSAGE.realtimeRefused(event.code),
						RealtimeUnsupportedReason.CONNECTION
					)
				: new RealtimeError(VOICE_ERROR_MESSAGE.realtimeClosedEarly);

		if (this.state === RealtimeState.OPEN) this.options.onError?.(error);

		this.fail(error);
	}

	private handleCompleted(transcript: string): void {
		this.completedText = joinPieces(this.completedText, cleanTranscript(transcript));
		this.itemText = '';
		this.inFlight = Math.max(0, this.inFlight - 1);
		this.options.onCompleted?.(this.text);

		if (this.inFlight === 0) {
			for (const wake of this.idleWaiters.splice(0)) wake();
		}

		// a live session completes once, on its final commit; a per-commit one once the last piece is in
		if (
			this.state === RealtimeState.FINISHING &&
			(this.live || (this.finalSent && this.inFlight === 0))
		) {
			this.resolveFinish();
		}
	}

	private handleError(message: string): void {
		if (this.state === RealtimeState.CONNECTING) {
			this.fail(
				VOICE_REALTIME_UNSUPPORTED_PATTERN.test(message)
					? new RealtimeUnsupportedError(message, RealtimeUnsupportedReason.MODEL)
					: new RealtimeError(message)
			);

			return;
		}

		const error = new RealtimeError(message);

		// a piece that failed gets no transcript: commit the next one, and let a finish go on
		if (this.inFlight > 0) {
			this.inFlight = 0;

			for (const wake of this.idleWaiters.splice(0)) wake();
		}

		this.options.onError?.(error);

		if (this.state === RealtimeState.FINISHING) this.fail(error);
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
						: VOICE_ERROR_MESSAGE.realtimeFailed
				);

				break;
		}
	}

	private handleSessionCreated(event: RealtimeServerEvent): void {
		if (this.state !== RealtimeState.CONNECTING) return;

		const rate = event.session?.audio?.input?.format?.rate;

		this.modelSampleRate = typeof rate === 'number' ? rate : null;
		this.live = event.session?.type === VOICE_REALTIME_SESSION_TRANSCRIPTION;
		this.clearTimer();

		// the rate of our chunks, and a transcription session for a model that would hold a conversation
		this.send({
			session: {
				audio: {
					input: {
						format: {
							rate: this.options.sampleRate,
							type: VOICE_REALTIME_AUDIO_FORMAT
						},
						...(this.options.language ? { transcription: { language: this.options.language } } : {})
					}
				},
				type: VOICE_REALTIME_SESSION_TRANSCRIPTION
			},
			type: VOICE_REALTIME_EVENTS.SESSION_UPDATE
		});

		this.state = RealtimeState.OPEN;

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
export function joinPieces(before: string, piece: string): string {
	if (!before) return piece;

	if (!piece) return before;

	return VOICE_TRAILING_SPACE_PATTERN.test(before) || VOICE_LEADING_SPACE_PATTERN.test(piece)
		? before + piece
		: `${before} ${piece}`;
}
