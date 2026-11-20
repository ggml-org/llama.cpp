import { VOICE_PAUSE_DEFAULTS } from './constants.js';
import { levelToDb } from './levels.js';
import { SpeechDetector } from './speech-detector.js';

/**
 * Finds the pauses in speech from the live input levels (0..1, see rmsToLevel), so a
 * recording can be transcribed piece by piece while it goes on, cut where the speaker
 * breathes rather than inside a word. A SpeechDetector tells speech, and quiet, from the
 * room's noise.
 */
export interface PauseDetectorOptions {
	/** Time between two levels, as the capture sends them */
	intervalMs: number;
	/** Silence after speech that makes a pause */
	pauseMs?: number;
	/** Speech needed before a pause counts */
	minSpeechMs?: number;
	/**
	 * A piece longer than this is cut at its next quiet moment, pause or not; half as
	 * long again and it is cut where it is, as with music or talk in the room no moment
	 * is ever quiet
	 */
	maxSegmentMs?: number;
}

export class PauseDetector {
	private readonly detector = new SpeechDetector();
	private readonly intervalMs: number;
	private readonly maxSegmentMs: number;
	private readonly minSpeechMs: number;
	private readonly pauseMs: number;
	private quietMs = 0;
	private segmentMs = 0;
	private speechMs = 0;

	/** Whether speech was heard since the last cut. */
	get heardSpeech(): boolean {
		return this.speechMs > 0;
	}

	constructor(options: PauseDetectorOptions) {
		this.intervalMs = options.intervalMs;
		this.pauseMs = options.pauseMs ?? VOICE_PAUSE_DEFAULTS.PAUSE_MS;
		this.minSpeechMs = options.minSpeechMs ?? VOICE_PAUSE_DEFAULTS.MIN_SPEECH_MS;
		this.maxSegmentMs = options.maxSegmentMs ?? VOICE_PAUSE_DEFAULTS.MAX_SEGMENT_MS;
	}

	/** Takes the newest level; true when this is a place to cut the recording. */
	push(level: number): boolean {
		const db = levelToDb(level);
		const loud = this.detector.push(db, this.intervalMs);

		this.segmentMs += this.intervalMs;

		if (loud) {
			this.speechMs += this.intervalMs;
			this.quietMs = 0;
		}

		const spoke = this.speechMs >= this.minSpeechMs;

		// no quiet moment came: the text must not wait for one that a noisy room never has
		if (spoke && this.segmentMs >= this.maxSegmentMs * VOICE_PAUSE_DEFAULTS.OVERTIME) {
			this.reset();

			return true;
		}

		if (loud || !this.detector.isQuiet(db)) return false;

		this.quietMs += this.intervalMs;

		const cut = spoke && (this.quietMs >= this.pauseMs || this.segmentMs >= this.maxSegmentMs);

		if (cut) this.reset();

		return cut;
	}

	/** Starts a new piece, e.g. after one was cut elsewhere. */
	reset(): void {
		this.quietMs = 0;
		this.segmentMs = 0;
		this.speechMs = 0;
	}
}
