// Guards the clipboard contract of the chat-form contenteditable:
// copy/cut expose the markdown SOURCE of the selection (each badge
// contributes its full `[name](file://...)` link) and pasting such
// markdown re-renders the badges.

import { describe, it, expect } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { tick } from 'svelte';
import ChatFormContenteditable from '$lib/components/app/chat/ChatForm/ChatFormContenteditable.svelte';

const SOURCE = 'hello [docs](file:///a/b) world';
const BADGE_SELECTOR = '[data-mention-badge="true"]';

function editableIn(container: HTMLElement): HTMLElement {
	const el = container.querySelector('[role="textbox"]');
	if (!(el instanceof HTMLElement)) throw new Error('contenteditable not rendered');
	return el;
}

function setSelection(root: HTMLElement, place: (range: Range, root: HTMLElement) => void) {
	const range = document.createRange();
	place(range, root);
	const selection = window.getSelection();
	if (!selection) throw new Error('no selection');
	selection.removeAllRanges();
	selection.addRange(range);
}

function clipboardEvent(type: 'copy' | 'cut' | 'paste', text = '') {
	const data = new DataTransfer();
	if (text) data.setData('text/plain', text);
	const event = new ClipboardEvent(type, { clipboardData: data, bubbles: true, cancelable: true });
	return { event, data };
}

describe('ChatFormContenteditable clipboard', () => {
	it('copy exposes the markdown source of the selection', async () => {
		const { container } = render(ChatFormContenteditable, { value: SOURCE });
		await tick();

		const root = editableIn(container);
		setSelection(root, (range) => range.selectNodeContents(root));

		const { event, data } = clipboardEvent('copy');
		root.dispatchEvent(event);

		expect(event.defaultPrevented).toBe(true);
		expect(data.getData('text/plain')).toBe(SOURCE);
	});

	it('cut exposes the markdown source and removes the slice', async () => {
		const { container } = render(ChatFormContenteditable, { value: SOURCE });
		await tick();

		const root = editableIn(container);
		setSelection(root, (range) => {
			const badge = root.querySelector(BADGE_SELECTOR);
			if (!badge) throw new Error('badge not rendered');
			range.setStartBefore(badge);
			range.setEndAfter(badge);
		});

		const { event, data } = clipboardEvent('cut');
		root.dispatchEvent(event);

		expect(event.defaultPrevented).toBe(true);
		expect(data.getData('text/plain')).toBe('[docs](file:///a/b)');
		expect(root.querySelector(BADGE_SELECTOR)).toBeNull();
		expect(root.textContent).toBe('hello  world');
	});

	it('paste of markdown mention links re-renders badges', async () => {
		const { container } = render(ChatFormContenteditable, { value: 'hello ' });
		await tick();

		const root = editableIn(container);
		root.focus();
		setSelection(root, (range) => {
			range.selectNodeContents(root);
			range.collapse(false);
		});

		const { event } = clipboardEvent('paste', '[docs](file:///a/b) world');
		root.dispatchEvent(event);
		await tick();

		expect(event.defaultPrevented).toBe(true);
		const badge = root.querySelector(BADGE_SELECTOR);
		expect(badge).not.toBeNull();
		expect(badge!.getAttribute('data-mention-name')).toBe('docs');
		expect(root.textContent).toContain('world');
	});

	it('paste without mention links keeps the DOM untouched', async () => {
		const { container } = render(ChatFormContenteditable, { value: 'hello ' });
		await tick();

		const root = editableIn(container);
		root.focus();
		setSelection(root, (range) => {
			range.selectNodeContents(root);
			range.collapse(false);
		});
		const firstChild = root.firstChild;

		const { event } = clipboardEvent('paste', 'plain text');
		root.dispatchEvent(event);
		await tick();

		expect(event.defaultPrevented).toBe(true);
		expect(root.querySelector(BADGE_SELECTOR)).toBeNull();
		// no rebuild: the live text node is the same instance
		expect(root.firstChild).toBe(firstChild);
	});
});
