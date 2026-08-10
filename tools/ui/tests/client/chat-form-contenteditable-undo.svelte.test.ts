// Guards the editing-key contract of the chat-form contenteditable:
// undo/redo is replayed from source snapshots (the token rebuilds destroy
// the native undo stack), and Tab is NOT intercepted (WCAG 2.1.2 no
// keyboard trap), matching the plain textarea.

import ChatFormContenteditableHarness from './components/ChatFormContenteditableHarness.svelte';
import { tick } from 'svelte';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const SOURCE = 'see [docs](file:///a/b)';

function editableIn(container: HTMLElement): HTMLElement {
	const el = container.querySelector('[role="textbox"]');

	if (!(el instanceof HTMLElement)) throw new Error('contenteditable not rendered');

	return el;
}

function type(root: HTMLElement, text: string, inputType = 'insertText') {
	root.appendChild(document.createTextNode(text));
	root.dispatchEvent(new InputEvent('input', { bubbles: true, inputType }));
}

function keydown(root: HTMLElement, init: KeyboardEventInit) {
	const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init });

	root.dispatchEvent(event);

	return event;
}

describe('ChatFormContenteditable undo/redo', () => {
	it('undoes and redoes an edit across a badge-containing buffer', async () => {
		const screen = render(ChatFormContenteditableHarness, { value: SOURCE });

		await tick();

		const root = editableIn(screen.container);

		type(root, ' more');
		await tick();
		expect(screen.component.getValue()).toBe(`${SOURCE} more`);

		const undoEvent = keydown(root, { ctrlKey: true, key: 'z' });

		await tick();
		expect(undoEvent.defaultPrevented).toBe(true);
		expect(screen.component.getValue()).toBe(SOURCE);

		const redoEvent = keydown(root, { ctrlKey: true, key: 'z', shiftKey: true });

		await tick();
		expect(redoEvent.defaultPrevented).toBe(true);
		expect(screen.component.getValue()).toBe(`${SOURCE} more`);
	});

	it('redoes with Ctrl+Y as well', async () => {
		const screen = render(ChatFormContenteditableHarness, { value: SOURCE });

		await tick();

		const root = editableIn(screen.container);

		type(root, ' more');
		await tick();
		keydown(root, { key: 'z', metaKey: true });
		await tick();
		expect(screen.component.getValue()).toBe(SOURCE);

		keydown(root, { ctrlKey: true, key: 'y' });
		await tick();
		expect(screen.component.getValue()).toBe(`${SOURCE} more`);
	});

	it('coalesces a typing burst into one undo step', async () => {
		const screen = render(ChatFormContenteditableHarness, { value: 'abc' });

		await tick();

		const root = editableIn(screen.container);

		type(root, 'd');
		type(root, 'e');
		await tick();
		expect(screen.component.getValue()).toBe('abcde');

		keydown(root, { ctrlKey: true, key: 'z' });
		await tick();
		expect(screen.component.getValue()).toBe('abc');
	});

	it('keeps a newline as its own undo step', async () => {
		const screen = render(ChatFormContenteditableHarness, { value: 'abc' });

		await tick();

		const root = editableIn(screen.container);

		type(root, 'd');
		type(root, '\n', 'insertLineBreak');
		await tick();
		expect(screen.component.getValue()).toBe('abcd\n');

		keydown(root, { ctrlKey: true, key: 'z' });
		await tick();
		expect(screen.component.getValue()).toBe('abcd');

		keydown(root, { ctrlKey: true, key: 'z' });
		await tick();
		expect(screen.component.getValue()).toBe('abc');
	});

	it('is a no-op when there is nothing to undo', async () => {
		const screen = render(ChatFormContenteditableHarness, { value: 'abc' });

		await tick();

		const root = editableIn(screen.container);
		const event = keydown(root, { ctrlKey: true, key: 'z' });

		await tick();

		expect(event.defaultPrevented).toBe(true);
		expect(screen.component.getValue()).toBe('abc');
	});

	it('abandons the redo branch after a fresh edit', async () => {
		const screen = render(ChatFormContenteditableHarness, { value: 'abc' });

		await tick();

		const root = editableIn(screen.container);

		type(root, 'd');
		await tick();
		keydown(root, { ctrlKey: true, key: 'z' });
		await tick();
		expect(screen.component.getValue()).toBe('abc');

		type(root, 'e');
		await tick();
		keydown(root, { ctrlKey: true, key: 'z', shiftKey: true });
		await tick();
		expect(screen.component.getValue()).toBe('abce');
	});
});

describe('ChatFormContenteditable Tab key', () => {
	it('does not trap Tab (focus can leave the editable)', async () => {
		const screen = render(ChatFormContenteditableHarness, { value: SOURCE });

		await tick();

		const root = editableIn(screen.container);
		const event = keydown(root, { key: 'Tab' });

		expect(event.defaultPrevented).toBe(false);
	});
});
