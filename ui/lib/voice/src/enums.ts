/** Voice engine enums: the states and reasons the engine reports. */

/** Where a VoiceCapture is. */
export enum VoiceCaptureState {
	IDLE = 'idle',
	RECORDING = 'recording',
	STARTING = 'starting',
	STOPPED = 'stopped'
}

/** Where a VoiceRecorder (voice/svelte) is. */
export enum VoiceRecorderStatus {
	IDLE = 'idle',
	RECORDING = 'recording',
	STARTING = 'starting',
	STOPPING = 'stopping'
}

/** Where a RealtimeTranscriber's socket is. */
export enum RealtimeState {
	CLOSED = 'closed',
	CONNECTING = 'connecting',
	FINISHING = 'finishing',
	IDLE = 'idle',
	OPEN = 'open'
}

/**
 * Why realtime transcription is not available: the model has no streaming audio encoder,
 * or the socket could not be opened.
 */
export enum RealtimeUnsupportedReason {
	CONNECTION = 'connection',
	MODEL = 'model'
}
