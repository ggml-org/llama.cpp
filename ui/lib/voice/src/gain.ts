import { VOICE_GAIN, VOICE_SAMPLE_RATE } from './constants.js';
import { rms, rmsToDb } from './levels.js';
import { floatToPcm16, pcm16ToFloat } from './pcm.js';
import { isSpeechLevel, SpeechDetector } from './speech-detector.js';

/**
 * Speech gain: brings a quiet microphone up to a level transcription models hear well.
 * Only speech is measured, never the room's silence (see SpeechDetector), so a pause does not pump the gain
 * up; the gain only ever raises the input (up to VOICE_GAIN.MAX), never lowers it, and
 * a soft limiter keeps the raised peaks from clipping.
 *
 * Two forms: SpeechGain follows the input block by block for what streams while
 * speaking, and normalizeSpeech() treats a finished recording in one pass, so its
 * first words get the same gain as the rest.
 */

/** What normalizeSpeech() did to a recording. */
export interface SpeechGainResult {
	/** The recording, raised, or the same samples when no speech was found or it was loud enough */
	samples: Int16Array;
	/** The gain applied, 1 when none */
	gain: number;
	/** The speech is quiet even at the highest gain: a sign the microphone is too far or too low */
	tooQuiet: boolean;
}

/** Gain that brings speech of this RMS to the target, within 1..max. */
export function speechGainFor(speechRms: number, max: number = VOICE_GAIN.MAX): number {
	if (speechRms <= 0) return 1;

	return Math.max(1, Math.min(max, VOICE_GAIN.TARGET_RMS / speechRms));
}

/** Whether speech of this RMS stays quiet even at the highest gain. */
export function isTooQuiet(speechRms: number): boolean {
	return speechRms > 0 && speechRms * VOICE_GAIN.MAX < VOICE_GAIN.TOO_QUIET_RMS;
}

/**
 * Soft limiter: samples up to the knee pass as they are, louder ones bend smoothly
 * towards full scale and never reach it.
 */
export function softLimit(sample: number): number {
	const magnitude = Math.abs(sample);

	if (magnitude <= VOICE_GAIN.KNEE) return sample;

	const room = 1 - VOICE_GAIN.KNEE;
	const bent = VOICE_GAIN.KNEE + room * Math.tanh((magnitude - VOICE_GAIN.KNEE) / room);

	return Math.sign(sample) * bent;
}

/** RMS of each frame of `samples`, frames of `frameLength` (the last one may be shorter). */
function frameLevels(samples: Float32Array, frameLength: number): number[] {
	const levels: number[] = [];

	for (let start = 0; start < samples.length; start += frameLength) {
		levels.push(rms(samples.subarray(start, Math.min(samples.length, start + frameLength))));
	}

	return levels;
}

/**
 * The loudness of the speech in a recording: frames that are speech against its noise
 * floor (its quietest tenth, see isSpeechLevel); their RMS together. 0 when nothing in it
 * is speech.
 */
export function measureSpeech(samples: Float32Array, sampleRate: number): number {
	const levels = frameLevels(
		samples,
		Math.max(1, Math.round((sampleRate * VOICE_GAIN.FRAME_MS) / 1000))
	);

	if (levels.length === 0) return 0;

	const sorted = [...levels].sort((a, b) => a - b);
	const floorDb = rmsToDb(sorted[Math.floor((sorted.length - 1) * VOICE_GAIN.FLOOR_QUANTILE)]);

	let power = 0;
	let count = 0;

	for (const level of levels) {
		if (!isSpeechLevel(rmsToDb(level), floorDb)) continue;

		power += level * level;
		count++;
	}

	return count > 0 ? Math.sqrt(power / count) : 0;
}

/** Raises a finished recording's speech to the target level in one pass; see SpeechGainResult. */
export function normalizeSpeech(
	samples: Int16Array,
	sampleRate: number,
	max: number = VOICE_GAIN.MAX
): SpeechGainResult {
	const floats = pcm16ToFloat(samples);
	const speech = measureSpeech(floats, sampleRate);
	const gain = speechGainFor(speech, max);
	const tooQuiet = isTooQuiet(speech);

	if (gain <= 1) return { gain: 1, samples, tooQuiet };

	for (let i = 0; i < floats.length; i++) floats[i] = softLimit(floats[i] * gain);

	return { gain, samples: floatToPcm16(floats), tooQuiet };
}

/**
 * Speech gain that follows the input as it streams. A SpeechDetector tells which blocks
 * are speech; a block of speech moves the speech level, and the
 * gain heads for what that level needs: it falls at once when speech gets louder, so
 * nothing clips, and rises over a few blocks when it gets quieter. Silence holds the
 * gain where it is.
 */
export class SpeechGain {
	/** The gain applied to the last block. */
	gain = 1;

	private readonly detector = new SpeechDetector();
	private speech = 0;

	/** Whether the speech heard so far stays quiet even at the highest gain. */
	get tooQuiet(): boolean {
		return isTooQuiet(this.speech);
	}

	/** `sampleRate` of the blocks, to follow the room at the same pace whatever their length. */
	constructor(
		private readonly max: number = VOICE_GAIN.MAX,
		private readonly sampleRate: number = VOICE_SAMPLE_RATE
	) {}

	/**
	 * Measures `block` and returns it raised by the gain, ramped from the previous block's
	 * gain so the change makes no click. `apply` false only measures, returning the block as is.
	 */
	process(block: Float32Array, apply = true): Float32Array {
		if (block.length === 0) return block;

		const level = rms(block);
		const isSpeech = this.detector.push(rmsToDb(level), (block.length / this.sampleRate) * 1000);
		const previous = this.gain;

		if (isSpeech) {
			this.speech =
				this.speech === 0 ? level : this.speech + (level - this.speech) * VOICE_GAIN.SPEECH_FOLLOW;

			const target = speechGainFor(this.speech, this.max);

			this.gain = target < this.gain ? target : this.gain + (target - this.gain) * VOICE_GAIN.RISE;
			// a block much louder than the speech so far must not clip waiting for the level to follow
			this.gain = Math.min(this.gain, Math.max(1, VOICE_GAIN.KNEE / level));
		}

		if (!apply || (previous === 1 && this.gain === 1)) return block;

		const out = new Float32Array(block.length);
		const step = (this.gain - previous) / block.length;

		for (let i = 0; i < block.length; i++) out[i] = softLimit(block[i] * (previous + step * i));

		return out;
	}
}
