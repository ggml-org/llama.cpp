/** The Voice input setting: what the microphone does with a model that understands audio. */
export enum VoiceInputMode {
	/** Send the voice to a model with audio input, dictate otherwise. */
	AUTO = 'auto',
	/** Always turn speech into text; only the text reaches the chat. */
	DICTATION = 'dictation'
}

/** What a recording becomes for the conversation's model, as the chat form resolves it. */
export enum VoiceMode {
	/** Speech becomes text in the input; no audio is attached. */
	DICTATION = 'dictation',
	/** No model can take audio, so the microphone has nothing to do. */
	UNAVAILABLE = 'unavailable',
	/** The clip is attached and sent as audio, with a transcript to read. */
	VOICE_MESSAGE = 'voice-message'
}

/** A model's part in voice input, as the model picker and the Models manager show it. */
export enum VoiceModelRole {
	/** The model that transcribes dictation for chat models that cannot hear */
	DICTATION = 'dictation',
	/** A speech-to-text model: for dictation, not for chat */
	TRANSCRIPTION = 'transcription',
	/** A chat model with audio input: it gets voice messages */
	VOICE = 'voice'
}
