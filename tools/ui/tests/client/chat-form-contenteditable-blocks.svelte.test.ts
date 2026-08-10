// Guards the newline contract of the chat-form contenteditable: browsers
// restructure the flat DOM on Enter (`<div>` wrappers, `<br>` shapes) and
// serialization must fold those back into `\n` so the emitted value never
// diverges from what is on screen.

import ChatFormContenteditableHarness from './components/ChatFormContenteditableHarness.svelte';
import { tick } from 'svelte';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const SOURCE = 'see [docs](file:///a/b) here';

function editableIn(container: HTMLElement): HTMLElement {
	const el = container.querySelector('[role="textbox"]');

	if (!(el instanceof HTMLElement)) throw new Error('contenteditable not rendered');

	return el;
}

function fireInput(root: HTMLElement) {
	root.dispatchEvent(new InputEvent('input', { bubbles: true }));
}

function setCaret(node: Node, offset: number) {
	const range = document.createRange();

	range.setStart(node, offset);
	range.setEnd(node, offset);
	const selection = window.getSelection();

	if (!selection) throw new Error('no selection');

	selection.removeAllRanges();
	selection.addRange(range);
}

describe('ChatFormContenteditable browser newline shapes', () => {
	it('serializes a Chromium Enter <div> wrapper as a newline', async () => {
		const screen = render(ChatFormContenteditableHarness, { value: SOURCE });

		await tick();

		const root = editableIn(screen.container);
		const div = document.createElement('div');

		div.textContent = 'second line';
		root.appendChild(div);
		fireInput(root);
		await tick();

		expect(screen.component.getValue()).toBe(`${SOURCE}\nsecond line`);
	});

	it('serializes a Firefox full <div> wrap as lines, badge included', async () => {
		const screen = render(ChatFormContenteditableHarness, { value: SOURCE });

		await tick();

		const root = editableIn(screen.container);
		const first = document.createElement('div');

		while (root.firstChild) first.appendChild(root.firstChild);
		const second = document.createElement('div');

		second.textContent = 'second line';
		root.appendChild(first);
		root.appendChild(second);
		fireInput(root);
		await tick();

		expect(screen.component.getValue()).toBe(`${SOURCE}\nsecond line`);
	});

	it('serializes a <br> as a newline', async () => {
		const screen = render(ChatFormContenteditableHarness, { value: 'here' });

		await tick();

		const root = editableIn(screen.container);

		root.appendChild(document.createElement('br'));
		root.appendChild(document.createTextNode('second line'));
		fireInput(root);
		await tick();

		expect(screen.component.getValue()).toBe('here\nsecond line');
	});

	it('ignores a trailing <br> (browser caret placeholder)', async () => {
		const screen = render(ChatFormContenteditableHarness, { value: 'abc' });

		await tick();

		const root = editableIn(screen.container);

		root.appendChild(document.createElement('br'));
		fireInput(root);
		await tick();

		expect(screen.component.getValue()).toBe('abc');
	});

	it('serializes one newline per empty-line <div><br></div>', async () => {
		const screen = render(ChatFormContenteditableHarness, { value: 'abc' });

		await tick();

		const root = editableIn(screen.container);

		for (let i = 0; i < 2; i++) {
			const div = document.createElement('div');

			div.appendChild(document.createElement('br'));
			root.appendChild(div);
		}
		fireInput(root);
		await tick();

		expect(screen.component.getValue()).toBe('abc\n\n');
	});

	it('treats a <div><br></div>-only buffer as empty for the placeholder', async () => {
		const screen = render(ChatFormContenteditableHarness, { value: 'abc' });

		await tick();

		const root = editableIn(screen.container);
		const div = document.createElement('div');

		div.appendChild(document.createElement('br'));
		root.replaceChildren(div);
		fireInput(root);
		await tick();

		expect(screen.component.getValue()).toBe('');
		expect(root.dataset.empty).toBe('true');
	});

	it('maps the caret across block boundaries in both directions', async () => {
		const screen = render(ChatFormContenteditableHarness, { value: 'abc\ndef' });

		await tick();

		// Rebuild into the Chromium block shape; the source is unchanged,
		// so no re-render fires.
		const root = editableIn(screen.container);
		const div = document.createElement('div');

		div.textContent = 'def';
		root.replaceChildren(document.createTextNode('abc'), div);
		fireInput(root);
		await tick();
		expect(screen.component.getValue()).toBe('abc\ndef');

		const divText = div.firstChild;

		if (!divText) throw new Error('div text missing');

		setCaret(divText, 2);
		expect(screen.component.getCaretOffset()).toBe(6);

		screen.component.setCaretOffset(6);
		const selection = window.getSelection();

		expect(selection?.anchorNode).toBe(divText);
		expect(selection?.anchorOffset).toBe(2);

		// The boundary newline itself: offset 3 is the end of "abc", offset
		// 4 the start of the "def" line.
		screen.component.setCaretOffset(4);
		expect(window.getSelection()?.anchorNode).toBe(divText);
		expect(window.getSelection()?.anchorOffset).toBe(0);

		screen.component.setCaretOffset(3);
		expect(window.getSelection()?.anchorNode).toBe(root.firstChild);
		expect(window.getSelection()?.anchorOffset).toBe(3);
	});
});
