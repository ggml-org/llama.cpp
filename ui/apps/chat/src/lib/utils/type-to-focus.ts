import { isIMEComposing } from './is-ime-composing';

const EDITABLE_SELECTOR =
	'input, textarea, select, [contenteditable]:not([contenteditable="false"])';
// widgets that take typed letters themselves (typeahead) or trap focus
const KEY_CAPTURING_SELECTOR =
	'[role="dialog"], [role="alertdialog"], [role="menu"], [role="listbox"]';

/**
 * Whether a keypress should move focus to the chat form so the character lands there:
 * a printable key typed while nothing that takes text has focus.
 */
export function shouldTypeToFocus(event: KeyboardEvent): boolean {
	if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return false;

	if (isIMEComposing(event)) return false;

	// a single character, not a named key; whitespace keeps scrolling or pressing the focused button
	if (event.key.length !== 1 || event.key.trim() === '') return false;

	const active = document.activeElement;

	if (active instanceof HTMLElement && active !== document.body) {
		if (active.closest(EDITABLE_SELECTOR) || active.closest(KEY_CAPTURING_SELECTOR)) return false;
	}

	return true;
}
