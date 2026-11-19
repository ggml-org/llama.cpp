/**
 * Voice input for the chat form.
 *
 * Resolves what the microphone does for the conversation's model (a voice message,
 * dictation, or nothing, see resolveVoiceMode) and runs a recording: live levels and
 * elapsed time, and on stop either a voice message attachment, whose transcript follows,
 * or the dictated text. Writing the text into the message is createDictation()'s; this
 * adds what is the chat form's own around it.
 * A voice message's live text shows as the empty message box's placeholder; focusing
 * the box makes it editable text there until the recording stops, when it goes with
 * the clip (edited, as its transcript).
 */

import {
	createDictation,
	describeMicrophoneError,
	MicrophoneWarmup,
	VoiceCaptureError,
	type VoiceRecorderStatus,
	type VoiceSessionOptions,
	type VoiceSessionResult
} from '@ggml-org/llama-voice';
import {
	SETTINGS_KEYS,
	VOICE_CLIP_NAME_PREFIX,
	VOICE_MODE_MESSAGE,
	VOICE_SEPARATE_MODEL_HINT_LOCALSTORAGE_KEY
} from '#lib/constants/index.ts';
import { MimeTypeAudio, VoiceMode } from '#lib/enums/index.ts';
import { VoiceService } from '#lib/services/voice.service.ts';
import {
	serverStore,
	settingsStore,
	voiceStore,
	voiceTranscriptsStore
} from '#lib/stores/index.ts';
import type { ChatUploadedFile } from '#lib/types/index.ts';
import { uuid } from '#lib/utils/index.ts';
import { voiceModeLabel } from '#lib/utils/voice-mode.ts';
import { onDestroy } from 'svelte';
import { toast } from 'svelte-sonner';

/** What the form needs of a VoiceRecorder; tests and stories pass a fake. */
export interface VoiceRecorderLike {
	readonly levels: number[];
	readonly live: boolean;
	readonly liveText: string;
	/** The speech stays quiet even at the highest gain; a recorder that cannot tell leaves it out */
	readonly quiet?: boolean;
	readonly status: VoiceRecorderStatus;
	start(options?: VoiceSessionOptions): Promise<void>;
	stop(): Promise<VoiceSessionResult>;
	cancel(): void;
}

/** Where a recording is: off, waiting for the microphone, recording, or waiting for dictated text. */
export type ChatFormVoiceState = 'idle' | 'starting' | 'recording' | 'transcribing';

interface UseChatFormVoiceOptions {
	getRecorder: () => VoiceRecorderLike | undefined;
	getValue: () => string;
	/** Writes the message as dictation changes it; not a user edit. */
	setValue: (value: string) => void;
	getCaretOffset: () => number;
	/** Puts the caret after the dictated text once it is final. */
	setCaretOffset: (offset: number) => void;
	addVoiceFile: (file: ChatUploadedFile) => void;
	/** Why the last start failed, for the microphone button; null once one worked. */
	onMicError: (message: string | null) => void;
}

const ELAPSED_TICK_MS = 200;

function readHintSeen(): boolean {
	try {
		return localStorage.getItem(VOICE_SEPARATE_MODEL_HINT_LOCALSTORAGE_KEY) === 'true';
	} catch {
		return false;
	}
}

function writeHintSeen(): void {
	try {
		localStorage.setItem(VOICE_SEPARATE_MODEL_HINT_LOCALSTORAGE_KEY, 'true');
	} catch {
		// private mode: the hint shows again next time
	}
}

function readAsDataUrl(blob: Blob): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();

		reader.onload = () => resolve(reader.result as string);
		reader.onerror = () => reject(reader.error);
		reader.readAsDataURL(blob);
	});
}

export function useChatFormVoice(options: UseChatFormVoiceOptions) {
	const config = $derived(settingsStore.config);
	const isRouter = $derived(serverStore.isRouterMode);
	// what the microphone does for the conversation's model, shared with the model lists
	const resolution = $derived(voiceStore.resolution);
	// a router with a model that hears keeps the button, so it can say where to pick it
	const anyAudioModel = $derived(voiceStore.anyAudioModel);

	let hintSeen = $state(readHintSeen());
	let state = $state<ChatFormVoiceState>('idle');
	let elapsedMs = $state(0);
	// the mode of the recording under way, fixed when it starts
	let recordingMode = $state<VoiceMode>(VoiceMode.UNAVAILABLE);
	let ticker: ReturnType<typeof setInterval> | undefined;
	// bumped when the dictation changes, so the getters that read it follow
	let dictationVersion = $state(0);
	// bumped by cancel, so a recording still starting or stopping is dropped
	let generation = 0;

	const warmup = new MicrophoneWarmup();
	const dictation = createDictation({
		getCaretOffset: options.getCaretOffset,
		getValue: options.getValue,
		onChange: () => dictationVersion++,
		onError: (error) =>
			toast.error(error instanceof Error ? error.message : 'Transcription failed'),
		setCaretOffset: options.setCaretOffset,
		setValue: options.setValue
	});
	const recorder = $derived(options.getRecorder());
	const unavailableReason = $derived(
		resolution.mode === VoiceMode.UNAVAILABLE ? resolution.reason : null
	);
	const hint = $derived(
		resolution.mode === VoiceMode.DICTATION && resolution.separateModel && !hintSeen
			? VOICE_MODE_MESSAGE.separateModelHint
			: null
	);

	function startTicker() {
		const startedAt = Date.now();

		elapsedMs = 0;
		ticker = setInterval(() => (elapsedMs = Date.now() - startedAt), ELAPSED_TICK_MS);
	}

	function stopTicker() {
		clearInterval(ticker);
		ticker = undefined;
	}

	/**
	 * A voice message's live text moves from the placeholder into the message, as text the
	 * user can edit, and what is said next follows it there.
	 */
	function adoptLiveText() {
		if (recordingMode !== VoiceMode.VOICE_MESSAGE || dictation.active) return;

		if (state !== 'starting' && state !== 'recording') return;

		dictation.begin({ at: options.getValue().length, keepsEdits: true });
		dictation.write(recorder?.liveText ?? '');
		options.setCaretOffset(dictation.end);
	}

	/** The session options of a recording started now: the settings, the server and the mode. */
	function wiringFor(mode: VoiceMode, model: string | null) {
		const transcribe =
			mode === VoiceMode.DICTATION || config[SETTINGS_KEYS.VOICE_TRANSCRIBE_MESSAGES] !== false;

		return VoiceService.sessionOptions(config, { isRouter, model, transcribe });
	}

	/**
	 * Opens the microphone while the pointer or the focus is on its button, if the page may
	 * already use it: by the click it has settled, so the first words are not quiet, and the
	 * recording starts at once. Let go shortly after the pointer leaves (unprepare).
	 */
	async function prepare() {
		if (!recorder || state !== 'idle' || resolution.mode === VoiceMode.UNAVAILABLE) return;

		const wiring = await wiringFor(resolution.mode, resolution.transcriptionModel);

		// a press that came meanwhile asked for its own microphone
		if (state === 'idle') warmup.open(wiring.capture);
	}

	function unprepare() {
		warmup.release();
	}

	async function start() {
		if (!recorder || state !== 'idle' || resolution.mode === VoiceMode.UNAVAILABLE) return;

		const mode = resolution.mode;
		const model = resolution.transcriptionModel;

		if (resolution.mode === VoiceMode.DICTATION && resolution.separateModel && !hintSeen) {
			hintSeen = true;
			writeHintSeen();
		}

		recordingMode = mode;
		state = 'starting';

		if (mode === VoiceMode.DICTATION) dictation.begin();

		const run = ++generation;

		try {
			const wiring = await wiringFor(mode, model);
			// the microphone the hover opened, when it is open for the same options
			const stream = await warmup.take(wiring.capture);

			await recorder.start({
				...wiring,
				capture: { ...wiring.capture, ...(stream ? { stream } : {}) },
				// a voice message's text reaches the message only once the user made it editable
				onText: (text) => dictation.write(text)
			});
		} catch (error) {
			if (run !== generation) return;

			state = 'idle';
			dictation.cancel();

			const message =
				error instanceof VoiceCaptureError ? error.message : describeMicrophoneError(error);

			options.onMicError(message);
			toast.error(message);

			return;
		}

		if (run !== generation) return;

		options.onMicError(null);
		state = 'recording';
		startTicker();
	}

	async function stop() {
		if (!recorder || state !== 'recording') return;

		const run = generation;
		const mode = recordingMode;

		stopTicker();

		let result: VoiceSessionResult;

		try {
			result = await recorder.stop();
		} catch (error) {
			console.error('Failed to stop the recording:', error);
			state = 'idle';

			return;
		}

		if (run !== generation) return;

		if (mode === VoiceMode.VOICE_MESSAGE) {
			// the clip carries the text from here: what the user edited is its transcript
			const taken = dictation.take();

			state = 'idle';
			await attachVoiceMessage(result, taken?.userEdited ? taken.text : null);

			return;
		}

		// with live text the message already shows it and can be edited, while the whole
		// recording is transcribed once more to refine it; otherwise wait for the text
		state = result.transcript && !result.liveText ? 'transcribing' : 'idle';
		await dictation.finish(result);

		if (run === generation) state = 'idle';
	}

	async function attachVoiceMessage(
		{ clip, transcript }: VoiceSessionResult,
		edited: string | null = null
	) {
		const name = `${VOICE_CLIP_NAME_PREFIX}${Date.now()}-${uuid().slice(0, 8)}.wav`;
		const file = new File([clip.blob], name, { type: MimeTypeAudio.WAV });

		if (transcript) voiceTranscriptsStore.track(name, transcript);

		if (edited !== null) voiceTranscriptsStore.setTranscript(name, edited);

		options.addVoiceFile({
			file,
			id: uuid(),
			name,
			preview: await readAsDataUrl(clip.blob),
			size: file.size,
			type: MimeTypeAudio.WAV,
			voice: { durationMs: clip.durationMs, peaks: clip.peaks }
		});
	}

	/** Drops the recording and any text it dictated. */
	function cancel() {
		generation++;
		stopTicker();
		recorder?.cancel();
		dictation.cancel();
		state = 'idle';
	}

	/** Follows an edit the user typed while dictation writes into the message. */
	function handleUserInput() {
		dictation.edited();
	}

	onDestroy(() => {
		warmup.drop();

		if (state !== 'idle') cancel();
	});

	return {
		adoptLiveText,
		cancel,
		get elapsedMs() {
			return elapsedMs;
		},
		handleUserInput,
		get hint() {
			return hint;
		},
		/** A recording (or its dictated text) is under way: the form shows the recording bar. */
		get isActive() {
			return state !== 'idle';
		},
		/** The dictated text shown is still being refined by a pass over the whole recording. */
		get isRefining() {
			void dictationVersion;

			return dictation.refining;
		},
		/** What the microphone button says it will do. */
		get label() {
			return voiceModeLabel(resolution);
		},
		get levels() {
			return recorder?.levels ?? [];
		},
		/** Text streaming while a voice message is recorded, shown under the waveform. */
		get liveCaption() {
			void dictationVersion;

			// once made editable, the text is in the message instead
			return recordingMode === VoiceMode.VOICE_MESSAGE && !dictation.active
				? (recorder?.liveText ?? '')
				: '';
		},
		get mode() {
			return state === 'idle' ? resolution.mode : recordingMode;
		},
		/** The pointer or the focus is on the microphone button: see prepare(). */
		prepare,
		/** The recording's speech is too quiet even at the highest gain: the form says so. */
		get quiet() {
			return state === 'recording' && (recorder?.quiet ?? false);
		},
		get resolution() {
			return resolution;
		},
		/** Resolves once the text in the message is final, at once when nothing is refining it. */
		settled(): Promise<void> {
			return dictation.settled();
		},
		/** Whether the form shows a microphone button at all. */
		get showButton() {
			return resolution.mode !== VoiceMode.UNAVAILABLE || anyAudioModel;
		},
		start,
		get state() {
			return state;
		},
		stop,
		get unavailableReason() {
			return unavailableReason;
		},
		/** The pointer or the focus left the microphone button without a press. */
		unprepare
	};
}
