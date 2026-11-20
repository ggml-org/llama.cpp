/** Where a recording is, as the recording bar shows it: waiting for the microphone, recording, or waiting for its text. */
export enum VoiceRecordingBarStatus {
	RECORDING = 'recording',
	STARTING = 'starting',
	WAITING = 'waiting'
}

/** Where a voice clip's transcript is: none asked for, on its way, there, or failed. */
export enum VoiceTranscriptStatus {
	DONE = 'done',
	ERROR = 'error',
	IDLE = 'idle',
	TRANSCRIBING = 'transcribing'
}

/** `SM` for a chip in the chat form, `MD` for a message bubble. */
export enum VoiceWaveformSize {
	MD = 'md',
	SM = 'sm'
}
