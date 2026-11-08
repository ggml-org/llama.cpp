export function isIMEComposing(event: KeyboardEvent) {
	// Check for IME composition using isComposing property and keyCode 229 (specifically for IME composition on Safari, which is notorious for not supporting KeyboardEvent.isComposing)
	// This prevents form submission when confirming IME word selection (e.g., Japanese/Chinese input)
	return event.isComposing || event.keyCode === 229;
}
