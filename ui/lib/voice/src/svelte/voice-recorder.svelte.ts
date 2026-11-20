import { VOICE_ERROR_MESSAGE, VOICE_LEVELS_LENGTH } from '../constants.js';
import { VoiceRecorderStatus } from '../enums.js';
import { startVoiceSession } from '../session.js';
import type { VoiceSession, VoiceSessionOptions, VoiceSessionResult } from '../types.js';

/**
 * Reactive voice session for components: `levels`, `liveText`, `live`, `quiet` and
 * `status` are runes that follow the running session.
 */
export class VoiceRecorder {
	/** Level history for the live waveform, 0..1, newest last */
	levels = $state.raw<number[]>(new Array(VOICE_LEVELS_LENGTH).fill(0));
	/** True once realtime is known to stream text while speaking */
	live = $state(false);
	/** Text the realtime socket streamed so far */
	liveText = $state('');
	/** True while the speech is too quiet even at the highest gain */
	quiet = $state(false);
	status = $state<VoiceRecorderStatus>(VoiceRecorderStatus.IDLE);

	// bumped by cancel(), so a start still waiting for the microphone drops its session
	private generation = 0;
	private session: VoiceSession | null = null;

	/** Drops the recording. */
	cancel(): void {
		this.generation++;
		this.session?.cancel();
		this.session = null;
		this.status = VoiceRecorderStatus.IDLE;
	}

	/**
	 * Starts a session; the callbacks in `options` still run.
	 * @throws VoiceCaptureError when the microphone does not start
	 */
	async start(options: VoiceSessionOptions = {}): Promise<void> {
		if (this.status !== VoiceRecorderStatus.IDLE) throw new Error(VOICE_ERROR_MESSAGE.recorderBusy);

		const capture = options.capture;
		const length =
			capture && 'levels' in capture
				? capture.levels.length
				: (capture?.levelsLength ?? VOICE_LEVELS_LENGTH);

		this.levels = new Array(length).fill(0);
		this.liveText = '';
		this.live = false;
		this.quiet = false;
		this.status = VoiceRecorderStatus.STARTING;

		const generation = ++this.generation;

		try {
			const session = await startVoiceSession({
				...options,
				onLevels: (levels) => {
					this.levels = levels as number[];
					options.onLevels?.(levels);
				},
				onQuiet: (quiet) => {
					this.quiet = quiet;
					options.onQuiet?.(quiet);
				},
				onRealtime: (live) => {
					this.live = live;
					options.onRealtime?.(live);
				},
				onText: (text) => {
					this.liveText = text;
					options.onText?.(text);
				}
			});

			if (generation !== this.generation) {
				session.cancel();

				return;
			}

			this.session = session;
			this.status = VoiceRecorderStatus.RECORDING;
		} catch (error) {
			if (generation === this.generation) this.status = VoiceRecorderStatus.IDLE;

			throw error;
		}
	}

	/** Ends the recording; see VoiceSessionResult. */
	async stop(): Promise<VoiceSessionResult> {
		const session = this.session;

		if (!session || this.status !== VoiceRecorderStatus.RECORDING) {
			throw new Error(VOICE_ERROR_MESSAGE.recorderIdle);
		}

		this.status = VoiceRecorderStatus.STOPPING;

		try {
			return await session.stop();
		} finally {
			this.session = null;
			this.status = VoiceRecorderStatus.IDLE;
		}
	}
}
