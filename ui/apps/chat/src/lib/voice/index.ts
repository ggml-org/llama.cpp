/**
 * Voice engine: microphone capture with live levels and PCM streaming, clip peaks,
 * and transcription over HTTP or llama-server's realtime socket. It imports no app
 * stores or components; the app passes in how to reach a server (VoiceServerContext).
 */

export { warmUpVoiceCapture } from './audio-context';
export { VoiceCapture } from './capture';
export * from './constants';
export {
	RealtimeError,
	RealtimeUnsupportedError,
	VoiceCaptureError,
	VoiceTranscriptionError
} from './errors';
export { LevelHistory, rms, rmsToLevel } from './levels';
export {
	base64ToBytes,
	bytesToBase64,
	concatPcm16,
	createResampler,
	floatToPcm16,
	pcm16ToBase64,
	pcm16ToFloat,
	type Resampler
} from './pcm';
export {
	listMicrophones,
	type Microphone,
	pickMicrophone,
	resolveMicrophone,
	VOICE_MICROPHONE_AUTO,
	VOICE_MICROPHONE_DEFAULT
} from './microphone';
export { computePeaks, computePeaksFromAudio, decodeAudio, mixToMono } from './peaks';
export { RealtimeTranscriber, realtimeUrl } from './realtime';
export { startVoiceSession } from './session';
export { transcribeClip } from './transcribe';
export type * from './types';
export { VoiceRecorder, type VoiceRecorderStatus } from './voice-recorder.svelte';
export { encodeWav } from './wav';
