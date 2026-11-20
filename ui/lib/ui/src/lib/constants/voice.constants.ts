import { VoiceWaveformSize } from '../enums/voice.enums.js';

/** A peak under this share of the clip's loudest one counts as a pause, not speech. */
export const VOICE_SPEECH_PEAK_FLOOR = 0.15;

/** The smallest bar, as a fraction of the waveform's height, so silence still draws a dotted line. */
export const MIN_BAR_HEIGHT = 0.12;

/** Whitespace, where a word ends. */
export const VOICE_WHITESPACE_PATTERN = /\s/;

/** A word with the whitespace around it, or whitespace alone. */
export const VOICE_WORD_PATTERN = /\s*\S+\s*|\s+/g;

/** Space between two waveform bars, in px. */
export const VOICE_BAR_GAP_PX = 2;

/** Width of a live waveform bar, in px. */
export const VOICE_LIVE_BAR_WIDTH_PX = 3;

/** How far the arrow keys move a clip's position. */
export const VOICE_SEEK_STEP_MS = 5000;

/** How long a new recording warning stays open above the row before it waits for hover or focus. */
export const VOICE_WARNING_SHOWN_MS = 6000;

/** What counts as a control inside a transcript reveal: a tap on one does not toggle the transcript. */
export const VOICE_INTERACTIVE_SELECTOR =
	'button, a, input, textarea, select, [role="slider"], [tabindex]';

/** Media query of a device without hover, where a tap opens the transcript. */
export const VOICE_NO_HOVER_QUERY = '(hover: none)';

/** Bar width and classes of a VoiceWaveform at each size. */
export const VOICE_WAVEFORM_SIZES = {
	[VoiceWaveformSize.SM]: {
		barWidthPx: 2,
		button: 'size-7',
		gap: 'gap-2',
		icon: 'size-icon-sm',
		time: 'min-w-8',
		track: 'h-5'
	},
	[VoiceWaveformSize.MD]: {
		barWidthPx: 3,
		button: 'size-8',
		gap: 'gap-3',
		icon: 'size-icon-md',
		time: 'min-w-9',
		track: 'h-7'
	}
} as const satisfies Record<
	VoiceWaveformSize,
	{ barWidthPx: number; button: string; gap: string; icon: string; time: string; track: string }
>;

/** What a VoiceWaveform is called when the app names it nothing else. */
export const VOICE_WAVEFORM_DEFAULT_LABEL = 'Voice message';

/** The recording bar's default labels, in English; an app passes its own. */
export const VOICE_RECORDING_BAR_LABELS = {
	cancel: 'Cancel recording',
	send: 'Stop and send',
	starting: 'Starting...',
	waiting: 'Transcribing...'
} as const;

/** What a transcript reveal says. */
export const VOICE_TRANSCRIPT_LABELS = {
	empty: 'No speech detected',
	error: 'Transcription unavailable',
	hide: 'Hide transcript',
	prefix: 'Transcript: ',
	show: 'Show transcript',
	transcribing: 'Transcribing...'
} as const;
