// Guards the clipboard contract of the chat-form contenteditable:
// copy/cut expose the markdown SOURCE of the selection (each badge
// contributes its full `[name](file://...)` link) and pasting such
// markdown re-renders the badges.

import { describe, it, expect, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { userEvent } from 'vitest/browser';
import { tick } from 'svelte';
import { rangeToTextOffset, serializeContent, textOffsetToRange } from '$lib/utils';
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

describe('ChatFormContenteditable code spans', () => {
	it('renders inline code from the initial value', async () => {
		const { container } = render(ChatFormContenteditable, { value: 'run `npm test` now' });
		await tick();

		const root = editableIn(container);
		const code = root.querySelector('code[data-code-token="inline"]');
		expect(code).not.toBeNull();
		expect(code!.textContent).toBe('`npm test`');
	});

	it('renders a fenced code block with a language', async () => {
		const source = 'before\n```js\nconst a = 1;\n```\nafter';
		const { container } = render(ChatFormContenteditable, { value: source });
		await tick();

		const root = editableIn(container);
		const code = root.querySelector('code[data-code-token="block"]');
		expect(code).not.toBeNull();
		expect(code!.textContent).toBe('```js\nconst a = 1;\n```');
	});

	it('copy exposes the markdown source of a selection spanning code', async () => {
		const source = 'run `npm test` now';
		const { container } = render(ChatFormContenteditable, { value: source });
		await tick();

		const root = editableIn(container);
		setSelection(root, (range) => range.selectNodeContents(root));

		const { event, data } = clipboardEvent('copy');
		root.dispatchEvent(event);

		expect(event.defaultPrevented).toBe(true);
		expect(data.getData('text/plain')).toBe(source);
	});

	it('paste of a code span renders the styled element', async () => {
		const { container } = render(ChatFormContenteditable, { value: 'run ' });
		await tick();

		const root = editableIn(container);
		root.focus();
		setSelection(root, (range) => {
			range.selectNodeContents(root);
			range.collapse(false);
		});

		const { event } = clipboardEvent('paste', '`npm test` now');
		root.dispatchEvent(event);
		await tick();

		expect(event.defaultPrevented).toBe(true);
		const code = root.querySelector('code[data-code-token="inline"]');
		expect(code).not.toBeNull();
		expect(code!.textContent).toBe('`npm test`');
		expect(root.textContent).toContain('now');
	});

	it('highlights a fenced block content and stays byte-exact', async () => {
		const source = '```js\nconst a = 1;\n```';
		const { container } = render(ChatFormContenteditable, { value: source });
		await tick();

		const root = editableIn(container);
		const code = root.querySelector('code[data-code-token="block"]');
		expect(code).not.toBeNull();
		expect(code!.querySelector('.hljs-keyword')).not.toBeNull();
		expect(code!.textContent).toBe(source);
	});

	it('does not highlight inline code', async () => {
		const { container } = render(ChatFormContenteditable, { value: 'run `const` now' });
		await tick();

		const root = editableIn(container);
		expect(root.querySelector('[class*="hljs-"]')).toBeNull();
	});
});

describe('ChatFormContenteditable code block escape hatches', () => {
	const BLOCK_SOURCE = '```js\nconst a = 1;\n```';
	const BLOCK_SELECTOR = 'code[data-code-token="block"]';

	function blockIn(root: HTMLElement): HTMLElement {
		const el = root.querySelector(BLOCK_SELECTOR);
		if (!(el instanceof HTMLElement)) throw new Error('code block not rendered');
		return el;
	}

	// Caret at the very start/end of the block's text (across highlight spans)
	function placeCaretInBlock(root: HTMLElement, where: 'start' | 'end') {
		const code = blockIn(root);
		const walker = document.createTreeWalker(code, NodeFilter.SHOW_TEXT);
		let target: Node | null = null;
		for (let n = walker.nextNode(); n; n = walker.nextNode()) {
			target = where === 'start' ? (target ?? n) : n;
		}
		if (!target) throw new Error('no text inside code block');
		setSelection(root, (range) => {
			range.setStart(target!, where === 'start' ? 0 : (target!.textContent ?? '').length);
			range.collapse(true);
		});
	}

	function caretContainer(): Node {
		const selection = window.getSelection();
		if (!selection || selection.rangeCount === 0) throw new Error('no selection');
		return selection.getRangeAt(0).startContainer;
	}

	it('pads a trailing code block with a br hatch that stays invisible to copy', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		// no permanent empty line above a leading block
		expect(root.firstChild).toBe(blockIn(root));
		expect(root.lastChild?.nodeName).toBe('BR');

		setSelection(root, (range) => range.selectNodeContents(root));
		const { event, data } = clipboardEvent('copy');
		root.dispatchEvent(event);

		expect(data.getData('text/plain')).toBe(BLOCK_SOURCE);
	});

	it('escapes a trailing code block with ArrowDown and types after it', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		root.focus();
		placeCaretInBlock(root, 'end');

		await userEvent.keyboard('{ArrowDown}');
		expect(blockIn(root).contains(caretContainer())).toBe(false);

		await userEvent.keyboard('x');
		await tick();

		expect(blockIn(root).textContent).toBe(BLOCK_SOURCE);
		// the DOM holds no separator newline (it would render as a
		// phantom empty line); serialization synthesizes it so the
		// markdown source keeps the text below the block
		expect(root.textContent).toBe(BLOCK_SOURCE + 'x');
		expect(serializeContent(root)).toBe(BLOCK_SOURCE + '\nx');
		// the stale trailing hatch is removed once real text follows the block
		expect(root.lastChild?.nodeName).not.toBe('BR');
	});

	it('escapes a leading code block with ArrowUp and types before it', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		root.focus();
		placeCaretInBlock(root, 'start');

		await userEvent.keyboard('{ArrowUp}');
		expect(blockIn(root).contains(caretContainer())).toBe(false);
		// the transient hatch line exists while the caret sits on it
		expect(root.firstChild?.nodeName).toBe('BR');

		await userEvent.keyboard('y');
		await tick();

		expect(blockIn(root).textContent).toBe(BLOCK_SOURCE);
		expect(root.textContent).toBe('y' + BLOCK_SOURCE);
		expect(serializeContent(root)).toBe('y\n' + BLOCK_SOURCE);
		// the typed text consumed the hatch
		expect(root.firstChild?.nodeName).not.toBe('BR');
	});

	it('escapes a leading code block with ArrowLeft from its first character', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		root.focus();
		placeCaretInBlock(root, 'start');

		await userEvent.keyboard('{ArrowLeft}');
		expect(blockIn(root).contains(caretContainer())).toBe(false);
		expect(root.firstChild?.nodeName).toBe('BR');
	});

	it('removes the transient leading hatch when the caret moves back into the block', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		root.focus();
		placeCaretInBlock(root, 'start');

		await userEvent.keyboard('{ArrowUp}');
		expect(root.firstChild?.nodeName).toBe('BR');

		await userEvent.keyboard('{ArrowDown}');
		await tick();

		expect(blockIn(root).contains(caretContainer())).toBe(true);
		expect(root.firstChild).toBe(blockIn(root));
	});

	it('extends the selection out of the block with Shift+ArrowDown', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		root.focus();
		placeCaretInBlock(root, 'end');

		await userEvent.keyboard('{Shift>}{ArrowDown}{/Shift}');

		const selection = window.getSelection();
		expect(selection).not.toBeNull();
		expect(selection!.isCollapsed).toBe(false);
		expect(blockIn(root).contains(selection!.getRangeAt(0).endContainer)).toBe(false);
	});

	it('line-separates text typed right after the closing fence', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		root.focus();
		placeCaretInBlock(root, 'end');

		// no arrow keys: the caret sits at the block's end edge, where the
		// post-rebuild restore lands it, and the typed text renders on the
		// line below the block
		await userEvent.keyboard('x');
		await tick();

		// the text stays on the caret's line in the DOM (no phantom empty
		// line); the source gets the separator newline
		expect(root.textContent).toBe(BLOCK_SOURCE + 'x');
		expect(serializeContent(root)).toBe(BLOCK_SOURCE + '\nx');
	});

	it('does not double the newline when Shift+Enter already added one', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		root.focus();
		placeCaretInBlock(root, 'end');

		await userEvent.keyboard('{Shift>}{Enter}{/Shift}');
		await userEvent.keyboard('x');
		await tick();

		expect(root.textContent).toBe(BLOCK_SOURCE + 'x');
		expect(serializeContent(root)).toBe(BLOCK_SOURCE + '\nx');
	});

	it('moves a caret stuck before the inserted newline onto the new line', async () => {
		const { container } = render(ChatFormContenteditable, {
			value: BLOCK_SOURCE + '\ntext after the code block'
		});
		await tick();

		const root = editableIn(container);
		root.focus();

		// post-break DOM some browsers produce: the inserted newline plus
		// the artificial trailing one, with the caret stuck BEFORE the
		// inserted one (visually at the end of the old line)
		root.appendChild(document.createTextNode('\n'));
		root.appendChild(document.createTextNode('\n'));
		setSelection(root, (range) => {
			range.setStart(root.childNodes[2], 0);
			range.collapse(true);
		});

		root.dispatchEvent(new InputEvent('input', { inputType: 'insertLineBreak', bubbles: true }));
		await tick();

		const selection = window.getSelection();
		expect(rangeToTextOffset(root, selection!.getRangeAt(0))).toBe(
			(BLOCK_SOURCE + '\ntext after the code block\n').length
		);
		expect(serializeContent(root)).toBe(BLOCK_SOURCE + '\ntext after the code block\n\n');
	});

	it('appends the artificial trailing newline when the browser did not add one', async () => {
		const { container } = render(ChatFormContenteditable, {
			value: BLOCK_SOURCE + '\ntext after the code block'
		});
		await tick();

		const root = editableIn(container);
		root.focus();

		// post-break DOM some browsers produce: a lone trailing \n (or a
		// <br> the hatch sync strips). Collapsed by the renderer, so the
		// caret looks stuck on the old line and the next typed character
		// would consume the newline.
		root.appendChild(document.createTextNode('\n'));
		setSelection(root, (range) => {
			range.setStart(root.childNodes[2], 1);
			range.collapse(true);
		});

		root.dispatchEvent(new InputEvent('input', { inputType: 'insertLineBreak', bubbles: true }));
		await tick();

		const selection = window.getSelection();
		expect(rangeToTextOffset(root, selection!.getRangeAt(0))).toBe(
			(BLOCK_SOURCE + '\ntext after the code block\n').length
		);
		expect(serializeContent(root)).toBe(BLOCK_SOURCE + '\ntext after the code block\n\n');
	});

	it('lands the caret on the new line with a single Shift+Enter after text below a block', async () => {
		const { container } = render(ChatFormContenteditable, {
			value: BLOCK_SOURCE + '\ntext after the code block'
		});
		await tick();

		const root = editableIn(container);
		root.focus();
		setSelection(root, (range) => {
			const text = root.childNodes[1];
			range.setStart(text, (text.textContent ?? '').length);
			range.collapse(true);
		});

		await userEvent.keyboard('{Shift>}{Enter}{/Shift}');
		await tick();

		const selection = window.getSelection();
		expect(rangeToTextOffset(root, selection!.getRangeAt(0))).toBe(
			(BLOCK_SOURCE + '\ntext after the code block\n').length
		);
		expect(serializeContent(root)).toBe(BLOCK_SOURCE + '\ntext after the code block\n\n');

		// the next typed character lands on the new line
		await userEvent.keyboard('x');
		await tick();
		expect(serializeContent(root)).toBe(BLOCK_SOURCE + '\ntext after the code block\nx');
	});

	it('lets Backspace at the text start move into the block without a source fight', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		root.focus();
		placeCaretInBlock(root, 'end');

		await userEvent.keyboard('{ArrowDown}');
		await userEvent.keyboard('create');
		await tick();
		expect(serializeContent(root)).toBe(BLOCK_SOURCE + '\ncreate');

		// Backspace at the start of the text line: the separator newline
		// is structural (synthesized while text follows the block), so
		// the caret just moves to the block's edge - nothing is re-added
		await userEvent.keyboard('{Home}');
		await userEvent.keyboard('{Backspace}');
		await tick();

		expect(serializeContent(root)).toBe(BLOCK_SOURCE + '\ncreate');
		expect(caretContainer()).toBe(root);
	});

	it('lets forward Delete eat the text after a block normally', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		root.focus();
		placeCaretInBlock(root, 'end');

		await userEvent.keyboard('{ArrowDown}');
		await userEvent.keyboard('create');
		await tick();

		await userEvent.keyboard('{Home}');
		await userEvent.keyboard('{Delete}');
		await tick();

		expect(serializeContent(root)).toBe(BLOCK_SOURCE + '\nreate');
	});

	it('renders text after a block without a phantom empty line', async () => {
		const { container } = render(ChatFormContenteditable, {
			value: BLOCK_SOURCE + '\nhello'
		});
		await tick();

		const root = editableIn(container);
		expect(root.textContent).toBe(BLOCK_SOURCE + 'hello');
		expect(serializeContent(root)).toBe(BLOCK_SOURCE + '\nhello');
	});

	it('keeps an intentional blank line after a block out of the separator', async () => {
		const { container } = render(ChatFormContenteditable, {
			value: BLOCK_SOURCE + '\n\nhello'
		});
		await tick();

		const root = editableIn(container);
		expect(root.textContent).toBe(BLOCK_SOURCE + '\nhello');
		expect(serializeContent(root)).toBe(BLOCK_SOURCE + '\n\nhello');
	});

	it('re-highlights while typing inside a block and keeps the caret', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		root.focus();

		// caret at the start of the block content (after the opening fence)
		setSelection(root, (range) => {
			const target = textOffsetToRange(root, 6);
			range.setStart(target.startContainer, target.startOffset);
			range.collapse(true);
		});

		await userEvent.keyboard('x');
		await tick();

		const code = blockIn(root);
		expect(serializeContent(root)).toBe('```js\nxconst a = 1;\n```');
		expect(code.textContent).toBe('```js\nxconst a = 1;\n```');
		expect(code.querySelector('.hljs-number')).not.toBeNull();

		const selection = window.getSelection();
		expect(code.contains(selection!.getRangeAt(0).startContainer)).toBe(true);
		expect(rangeToTextOffset(root, selection!.getRangeAt(0))).toBe(7);
	});
});

describe('ChatFormContenteditable Enter in code blocks', () => {
	const BLOCK_SOURCE = '```js\nconst a = 1;\n```';

	it('adds a line instead of submitting on plain Enter inside a block', async () => {
		const onKeydown = vi.fn();
		const { container } = render(ChatFormContenteditable, {
			value: BLOCK_SOURCE,
			onKeydown
		});
		await tick();

		const root = editableIn(container);
		root.focus();

		// caret at the start of the block content (after the opening fence)
		setSelection(root, (range) => {
			const target = textOffsetToRange(root, 6);
			range.setStart(target.startContainer, target.startOffset);
			range.collapse(true);
		});

		await userEvent.keyboard('{Enter}');
		await tick();

		// consumed locally: the parent's submit handler never sees it
		expect(onKeydown).not.toHaveBeenCalled();
		expect(serializeContent(root)).toBe('```js\n\nconst a = 1;\n```');

		const code = root.querySelector('code[data-code-token="block"]');
		const selection = window.getSelection();
		expect(code!.contains(selection!.getRangeAt(0).startContainer)).toBe(true);
		expect(rangeToTextOffset(root, selection!.getRangeAt(0))).toBe(7);
	});

	it('adds a line after a still-open fence (no closing ``` yet)', async () => {
		const onKeydown = vi.fn();
		const { container } = render(ChatFormContenteditable, {
			value: '```js\nconst a = 1;',
			onKeydown
		});
		await tick();

		const root = editableIn(container);
		root.focus();

		// caret at the start of the block content (after the opening fence)
		setSelection(root, (range) => {
			const target = textOffsetToRange(root, 6);
			range.setStart(target.startContainer, target.startOffset);
			range.collapse(true);
		});

		await userEvent.keyboard('{Enter}');
		await tick();

		expect(onKeydown).not.toHaveBeenCalled();
		expect(serializeContent(root)).toBe('```js\n\nconst a = 1;');
	});

	it('forwards plain Enter to the parent when the caret is outside a block', async () => {
		const onKeydown = vi.fn();
		const { container } = render(ChatFormContenteditable, {
			value: BLOCK_SOURCE + '\nafter',
			onKeydown
		});
		await tick();

		const root = editableIn(container);
		root.focus();
		setSelection(root, (range) => {
			range.selectNodeContents(root);
			range.collapse(false);
		});

		await userEvent.keyboard('{Enter}');

		expect(onKeydown).toHaveBeenCalledTimes(1);
		expect(onKeydown.mock.calls[0][0].defaultPrevented).toBe(false);
	});

	it('forwards plain Enter on the trailing hatch line after a block', async () => {
		const onKeydown = vi.fn();
		const { container } = render(ChatFormContenteditable, {
			value: BLOCK_SOURCE,
			onKeydown
		});
		await tick();

		const root = editableIn(container);
		root.focus();
		// root-level caret between the block and its trailing br hatch
		setSelection(root, (range) => {
			range.setStart(root, 1);
			range.collapse(true);
		});

		await userEvent.keyboard('{Enter}');

		expect(onKeydown).toHaveBeenCalledTimes(1);
	});

	it('forwards Ctrl+Enter inside a block so explicit submit survives', async () => {
		const onKeydown = vi.fn();
		const { container } = render(ChatFormContenteditable, {
			value: BLOCK_SOURCE,
			onKeydown
		});
		await tick();

		const root = editableIn(container);
		root.focus();
		setSelection(root, (range) => {
			const target = textOffsetToRange(root, 6);
			range.setStart(target.startContainer, target.startOffset);
			range.collapse(true);
		});

		await userEvent.keyboard('{Control>}{Enter}{/Control}');

		expect(onKeydown).toHaveBeenCalledWith(
			expect.objectContaining({ key: 'Enter', ctrlKey: true })
		);
		expect(serializeContent(root)).toBe(BLOCK_SOURCE);
	});

	it('forwards Enter inside an inline code span', async () => {
		const onKeydown = vi.fn();
		const { container } = render(ChatFormContenteditable, {
			value: 'run `npm test` now',
			onKeydown
		});
		await tick();

		const root = editableIn(container);
		root.focus();
		const code = root.querySelector('code[data-code-token="inline"]')!;
		setSelection(root, (range) => {
			range.setStart(code.firstChild!, 3);
			range.collapse(true);
		});

		await userEvent.keyboard('{Enter}');

		expect(onKeydown).toHaveBeenCalledTimes(1);
	});
});
