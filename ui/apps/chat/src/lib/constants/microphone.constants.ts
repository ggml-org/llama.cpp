/** Why the chat form's microphone can't record, as its button and toast tell the user. */
export const MICROPHONE_UNAVAILABLE_MESSAGE = {
	busy: 'The microphone is in use by another app.',
	denied: "Microphone access was denied. Allow it in the browser's site settings to record.",
	failed: "The microphone couldn't start.",
	insecure:
		'The microphone only works over HTTPS or on localhost. Open this page over HTTPS to record.',
	notFound: 'No microphone was found.',
	unsupported: "This browser can't record audio."
} as const;
