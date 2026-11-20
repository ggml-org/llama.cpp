/**
 * Llama Voice: microphone capture with live levels and PCM streaming, clip peaks,
 * transcription over HTTP or llama-server's realtime socket, and dictation into a text
 * field. It knows no app: the app passes in how to reach a server (VoiceServerContext).
 * Plain TypeScript: the Svelte adapters (VoiceRecorder) are in @ggml-org/llama-voice/svelte.
 */

export { warmUpVoiceCapture } from './audio-context.js';
export { VoiceCapture } from './capture.js';
export {
	VOICE_LEVELS_INTERVAL_MS,
	VOICE_LEVELS_LENGTH,
	VOICE_MICROPHONE_CONSTRAINTS,
	VOICE_MICROPHONE_MESSAGE,
	VOICE_PEAKS_COUNT,
	VOICE_SAMPLE_RATE,
	VOICE_WAV_MIME
} from './constants.js';
export {
	type BeginDictationOptions,
	createDictation,
	type Dictation,
	type DictationOptions,
	type DictationResult,
	type DictationTarget
} from './dictation.js';
export { DictationSpan } from './dictation-span.js';
export { RealtimeUnsupportedReason, VoiceCaptureState, VoiceRecorderStatus } from './enums.js';
export {
	RealtimeError,
	RealtimeUnsupportedError,
	VoiceCaptureError,
	VoiceTranscriptionError
} from './errors.js';
export { SpeechGain } from './gain.js';
export { levelToDb } from './levels.js';
export {
	listMicrophones,
	type Microphone,
	microphoneRequest,
	pickMicrophone,
	resolveMicrophone,
	VOICE_MICROPHONE_AUTO,
	VOICE_MICROPHONE_DEFAULT
} from './microphone.js';
export { PauseDetector, type PauseDetectorOptions } from './pauses.js';
export { base64ToBytes } from './pcm.js';
export { computePeaks, computePeaksFromAudio } from './peaks.js';
export { RealtimeTranscriber } from './realtime.js';
export { startVoiceSession } from './session.js';
export { SpeechDetector, type SpeechDetectorOptions } from './speech-detector.js';
export {
	describeMicrophoneError,
	isVoiceCaptureSupported,
	microphoneUnavailableReason
} from './support.js';
export { transcribeClip } from './transcribe.js';
export { cleanTranscript } from './transcript.js';
export type * from './types.js';
export { encodeWav } from './wav.js';
export { MicrophoneWarmup } from './warm-microphone.js';
