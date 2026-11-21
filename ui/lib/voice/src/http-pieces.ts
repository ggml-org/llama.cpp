import { concatPcm16 } from './pcm.js';
import { joinPieces } from './realtime.js';
import { encodeWav } from './wav.js';

export interface HttpPieceTranscriberOptions {
	/** Transcribes one piece, e.g. with transcribeClip() */
	transcribe: (wav: Blob, signal: AbortSignal) => Promise<string>;
	/** Rate of the PCM chunks given to append() */
	sampleRate: number;
	/** The whole text so far, each time a piece adds to it */
	onText?: (text: string) => void;
}

/**
 * Live text without a realtime session: for a model that hears but cannot hold one
 * (Qwen3-ASR), or a server a browser websocket cannot reach. The recording is cut into
 * pieces at the speaker's pauses (see cut()), each sent to `/v1/audio/transcriptions`
 * as a WAV of its own, one at a time, and their text joined in order.
 */
export class HttpPieceTranscriber {
	/** The pieces' text so far */
	text = '';

	private chunks: Int16Array[] = [];
	private closed = false;
	private inFlight: AbortController | null = null;
	private length = 0;
	private readonly options: HttpPieceTranscriberOptions;

	/**
	 * Whether a piece can be cut now: audio since the last one, and none still being
	 * transcribed, so pieces come back in order; meanwhile the next one grows.
	 */
	get canCut(): boolean {
		return !this.closed && this.inFlight === null && this.length > 0;
	}

	/** Takes a PCM16 mono chunk at `options.sampleRate`. */
	append(chunk: Int16Array): void {
		if (this.closed || chunk.length === 0) return;

		this.chunks.push(chunk);
		this.length += chunk.length;
	}

	/** Drops what is pending and stops a piece being transcribed. */
	close(): void {
		this.closed = true;
		this.inFlight?.abort();
		this.inFlight = null;
		this.chunks = [];
		this.length = 0;
	}

	constructor(options: HttpPieceTranscriberOptions) {
		this.options = options;
	}

	/** Ends the current piece and transcribes it; see canCut. */
	cut(): void {
		if (!this.canCut) return;

		const samples = concatPcm16(this.chunks);

		this.chunks = [];
		this.length = 0;

		const controller = new AbortController();

		this.inFlight = controller;
		this.options
			.transcribe(encodeWav(samples, this.options.sampleRate), controller.signal)
			.then(
				(piece) => {
					if (controller.signal.aborted || !piece.trim()) return;

					this.text = joinPieces(this.text, piece.trim());
					this.options.onText?.(this.text);
				},
				(error) => {
					if (!controller.signal.aborted) console.warn('Transcribing a piece failed:', error);
				}
			)
			.finally(() => {
				if (this.inFlight === controller) this.inFlight = null;
			});
	}
}
