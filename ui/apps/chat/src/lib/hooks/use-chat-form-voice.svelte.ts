/**
 * Voice input for the chat form.
 *
 * Resolves what the microphone does for the conversation's model (a voice message,
 * dictation, or nothing, see resolveVoiceMode) and runs a recording: live levels and
 * elapsed time, dictated text written into the message at the caret as it streams, and
 * on stop either a voice message attachment, whose transcript follows, or the text.
 * A voice message's live text shows as the empty message box's placeholder; focusing
 * the box makes it editable text there until the recording stops, when it goes with
 * the clip (edited, as its transcript).
 */

import {
	SETTINGS_KEYS,
	VOICE_CLIP_NAME_PREFIX,
	VOICE_MODE_MESSAGE,
	VOICE_SEPARATE_MODEL_HINT_LOCALSTORAGE_KEY
} from '#lib/constants/index.ts';
import { MimeTypeAudio, VoiceInputMode, VoiceMode } from '#lib/enums/index.ts';
import { VoiceService } from '#lib/services/voice.service.ts';
import {
	modelsStore,
	serverStore,
	settingsStore,
	voiceTranscriptsStore
} from '#lib/stores/index.ts';
import type { ChatUploadedFile } from '#lib/types/index.ts';
import { DictationSpan } from '#lib/utils/dictation-span.ts';
import { uuid } from '#lib/utils/index.ts';
import { resolveVoiceMode, voiceModeLabel } from '#lib/utils/voice-mode.ts';
import {
	describeMicrophoneError,
	VoiceCaptureError,
	type VoiceRecorderStatus,
	type VoiceSessionOptions,
	type VoiceSessionResult
} from '#lib/voice/index.ts';
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
	const chatModel = $derived(modelsStore.activeModelId);
	const transcriptionSetting = $derived(
		String(config[SETTINGS_KEYS.VOICE_TRANSCRIPTION_MODEL] ?? '')
	);
	const hasAudio = (model: string | null) => {
		void modelsStore.props.cacheVersion;

		return model ? modelsStore.props.modelSupportsAudio(model) : false;
	};
	// what Automatic transcribes with when the chat model can't hear: one already loaded
	const loadedAudioModel = $derived.by(() => {
		void modelsStore.props.cacheVersion;

		if (!isRouter) return null;

		return (
			modelsStore.models.find(
				(option) => hasAudio(option.model) && modelsStore.isModelLoaded(option.model)
			)?.model ?? null
		);
	});
	const resolution = $derived(
		resolveVoiceMode({
			chatModel,
			chatModelHasAudio: hasAudio(chatModel),
			isRouter,
			loadedAudioModel: loadedAudioModel,
			setting: String(config[SETTINGS_KEYS.VOICE_INPUT_MODE] ?? VoiceInputMode.AUTO),
			transcriptionModel: transcriptionSetting,
			transcriptionModelHasAudio: hasAudio(transcriptionSetting || null)
		})
	);
	// a router with a model that hears keeps the button, so it can say where to pick it
	const anyAudioModel = $derived.by(() => {
		void modelsStore.props.cacheVersion;

		return isRouter && modelsStore.models.some((option) => hasAudio(option.model));
	});

	let hintSeen = $state(readHintSeen());
	let state = $state<ChatFormVoiceState>('idle');
	let elapsedMs = $state(0);
	// the mode of the recording under way, fixed when it starts
	let recordingMode = $state<VoiceMode>(VoiceMode.UNAVAILABLE);
	let ticker: ReturnType<typeof setInterval> | undefined;
	// state: a voice message's text leaves the placeholder once it is in the message
	let span = $state<DictationSpan | null>(null);
	// the pass over the whole recording that follows live text, while it is still out
	let refining = $state<Promise<void> | null>(null);
	let lastValue = '';
	// bumped by cancel, so a transcript still on its way is dropped
	let generation = 0;

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

	function writeDictation(text: string) {
		if (!span) return;

		const next = span.apply(options.getValue(), text);

		lastValue = next;
		options.setValue(next);
	}

	/**
	 * A voice message's live text moves from the placeholder into the message, as text the
	 * user can edit, and what is said next follows it there.
	 */
	function adoptLiveText() {
		if (recordingMode !== VoiceMode.VOICE_MESSAGE || span) return;

		if (state !== 'starting' && state !== 'recording') return;

		lastValue = options.getValue();
		span = new DictationSpan(lastValue, lastValue.length);
		span.keepsEdits = true;
		writeDictation(recorder?.liveText ?? '');
		options.setCaretOffset(span.end);
	}

	async function start() {
		if (!recorder || state !== 'idle' || resolution.mode === VoiceMode.UNAVAILABLE) return;

		const mode = resolution.mode;
		const model = resolution.transcriptionModel;
		const transcribe =
			mode === VoiceMode.DICTATION || config[SETTINGS_KEYS.VOICE_TRANSCRIBE_MESSAGES] !== false;

		if (resolution.mode === VoiceMode.DICTATION && resolution.separateModel && !hintSeen) {
			hintSeen = true;
			writeHintSeen();
		}

		recordingMode = mode;
		state = 'starting';
		span = null;

		if (mode === VoiceMode.DICTATION) {
			lastValue = options.getValue();
			span = new DictationSpan(lastValue, options.getCaretOffset());
		}

		const run = ++generation;

		try {
			const wiring = await VoiceService.sessionOptions(config, { isRouter, model, transcribe });

			await recorder.start({
				...wiring,
				// a voice message's text reaches the message only once the user made it editable
				onText: (text) => writeDictation(text)
			});
		} catch (error) {
			if (run !== generation) return;

			state = 'idle';
			span = null;

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
			const edited = takeLiveText();

			state = 'idle';
			await attachVoiceMessage(result, edited);

			return;
		}

		await finishDictation(result, run);
	}

	/**
	 * Takes a voice message's text back out of the message once it stops: the clip carries
	 * it. Returns the text when the user edited it, which then is the clip's transcript.
	 */
	function takeLiveText(): string | null {
		const taken = span;

		span = null;

		if (!taken || taken.isEmpty) return null;

		const previous = options.getValue();
		const text = taken.text(previous).trim();

		lastValue = taken.remove(previous);
		options.setValue(lastValue);

		return taken.userEdited ? text : null;
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

	async function finishDictation({ liveText, transcript }: VoiceSessionResult, run: number) {
		const dictated = span;

		if (!transcript || !dictated) {
			state = 'idle';

			return;
		}

		// with live text the message already shows it and can be edited, while the whole
		// recording is transcribed once more to refine it; otherwise wait for the text
		if (!liveText) state = 'transcribing';
		else state = 'idle';

		const refined = (async () => {
			try {
				const text = await transcript;

				if (run !== generation) return;

				if (text.trim() || !liveText) writeDictation(text);
			} catch (error) {
				if (run !== generation) return;

				toast.error(error instanceof Error ? error.message : 'Transcription failed');
			}
		})();

		if (liveText) refining = refined;

		await refined;

		if (refining === refined) refining = null;

		if (run !== generation) return;

		state = 'idle';
		span = null;

		if (!dictated.isEmpty) options.setCaretOffset(dictated.end);
	}

	/** Drops the recording and any text it dictated. */
	function cancel() {
		generation++;
		stopTicker();
		recorder?.cancel();

		if (span) {
			const next = span.remove(options.getValue());

			options.setValue(next);
			span = null;
		}

		state = 'idle';
	}

	/** Follows an edit the user typed while dictation writes into the message. */
	function handleUserInput() {
		const next = options.getValue();

		span?.edited(lastValue, next);
		lastValue = next;
	}

	onDestroy(() => {
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
			return refining !== null;
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
			// once made editable, the text is in the message instead
			return recordingMode === VoiceMode.VOICE_MESSAGE && !span ? (recorder?.liveText ?? '') : '';
		},
		get mode() {
			return state === 'idle' ? resolution.mode : recordingMode;
		},
		/** The recording's speech is too quiet even at the highest gain: the form says so. */
		get quiet() {
			return state === 'recording' && (recorder?.quiet ?? false);
		},
		get resolution() {
			return resolution;
		},
		/** Resolves once the text in the message is final, at once when nothing is refining it. */
		settled(): Promise<void> {
			return refining ?? Promise.resolve();
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
		}
	};
}
