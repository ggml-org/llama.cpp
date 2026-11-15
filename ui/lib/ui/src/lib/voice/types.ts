/** Where a voice clip's transcript is: none asked for, on its way, there, or failed. */
export type VoiceTranscriptStatus = 'idle' | 'transcribing' | 'done' | 'error';

/** `sm` for a chip in the chat form, `md` for a message bubble. */
export type VoiceWaveformSize = 'sm' | 'md';
