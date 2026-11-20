import { VOICE_SPEECH } from './constants.js';

/**
 * Tells speech from the room it is said in. Levels are in dBFS; the noise floor follows
 * the room (it drops to any quieter level at once and creeps back up through louder
 * ones), and a level clearly above it, and above the absolute minimum, is speech. So a
 * quiet and a loud microphone both work, and so does a room with a fan running.
 *
 * Everything in the engine that needs to know whether someone is speaking asks one of
 * these: the pause detector that cuts a recording, the gain that raises quiet speech, and
 * a feature listening for speech over a reply. Tuning VOICE_SPEECH tunes them all.
 */
export interface SpeechDetectorOptions {
	/** The room's noise level in dBFS when it is known already, so speech from the first level counts */
	floorDb?: number;
	/** How far above the floor a level is speech, in dB; higher for a stricter listener */
	speechDb?: number;
	/** How far above the floor a level still is quiet, in dB */
	quietDb?: number;
}

/** Whether a level is speech, given the room's noise floor (both in dBFS). */
export function isSpeechLevel(
	db: number,
	floorDb: number,
	speechDb: number = VOICE_SPEECH.ABOVE_FLOOR_DB
): boolean {
	return db > VOICE_SPEECH.MIN_DB && db > floorDb + speechDb;
}

export class SpeechDetector {
	/** The room's noise level in dBFS, as heard so far. */
	floorDb: number;

	private readonly quietDb: number;
	private readonly speechDb: number;

	constructor(options: SpeechDetectorOptions = {}) {
		// full scale until the first level, which then sets it
		this.floorDb = options.floorDb ?? 0;
		this.quietDb = options.quietDb ?? VOICE_SPEECH.QUIET_ABOVE_FLOOR_DB;
		this.speechDb = options.speechDb ?? VOICE_SPEECH.ABOVE_FLOOR_DB;
	}

	/** Whether `db` is quiet against the floor: the room, or a breath between words. */
	isQuiet(db: number): boolean {
		return db <= this.floorDb + this.quietDb;
	}

	/** Whether `db` is speech against the floor. */
	isSpeech(db: number): boolean {
		return isSpeechLevel(db, this.floorDb, this.speechDb);
	}

	/**
	 * Takes the level of the newest `ms` of input, follows the floor with it, and returns
	 * whether it is speech.
	 */
	push(db: number, ms: number): boolean {
		this.floorDb = Math.min(db, this.floorDb + (VOICE_SPEECH.FLOOR_RISE_DB_PER_S * ms) / 1000);

		return this.isSpeech(db);
	}
}
