/**
 * Finds the pauses in speech from the live input levels (0..1, see rmsToLevel), so a
 * recording can be transcribed piece by piece while it goes on, cut where the speaker
 * breathes rather than inside a word. Levels are judged against the room's noise
 * floor, which it follows, so a quiet or a loud microphone both work.
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

/** How far above the noise floor a level is speech, and below which it is quiet. */
const SPEECH_ABOVE_FLOOR = 0.15;
const QUIET_ABOVE_FLOOR = 0.08;
/** How fast the floor rises back after it dropped to a quieter level, per level. */
const FLOOR_RISE = 0.002;
/** How far past its longest length a piece waits for a quiet moment before it is cut anyway. */
const OVERTIME = 1.5;

export class PauseDetector {
	private floor = 1;
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
		this.pauseMs = options.pauseMs ?? 500;
		this.minSpeechMs = options.minSpeechMs ?? 300;
		this.maxSegmentMs = options.maxSegmentMs ?? 10_000;
	}

	/** Takes the newest level; true when this is a place to cut the recording. */
	push(level: number): boolean {
		this.floor = Math.min(level, this.floor + FLOOR_RISE);
		this.segmentMs += this.intervalMs;

		const loud = level > this.floor + SPEECH_ABOVE_FLOOR;

		if (loud) {
			this.speechMs += this.intervalMs;
			this.quietMs = 0;
		}

		const spoke = this.speechMs >= this.minSpeechMs;

		// no quiet moment came: the text must not wait for one that a noisy room never has
		if (spoke && this.segmentMs >= this.maxSegmentMs * OVERTIME) {
			this.reset();

			return true;
		}

		if (loud || level > this.floor + QUIET_ABOVE_FLOOR) return false;

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
