// Guards the chat form input contract: a live-preview markdown editor whose value
// stays the exact markdown source, with mention links shown as badges while their
// block is edited and every block rendered once the editor loses focus. The form's
// key and paste handlers run first and win when they handle the event.

import { EditorView } from '@codemirror/view';
import ChatFormInput from '$lib/components/app/chat/ChatForm/ChatFormInput/ChatFormInput.svelte';
import { tick } from 'svelte';
import { describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const MENTION = '[notes.md](file:///tmp/notes.md)';

function editorIn(container: HTMLElement): EditorView {
	const el = container.querySelector<HTMLElement>('.cm-content');
	const view = el && EditorView.findFromDOM(el);

	if (!view) throw new Error('editor not rendered');

	return view;
}

async function settle() {
	await tick();
	await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
}

describe('ChatFormInput', () => {
	it('keeps the typed markdown byte-exact in the value', async () => {
		const screen = render(ChatFormInput, { value: '' });
		const editor = editorIn(screen.container);

		await userEvent.click(editor.contentDOM);
		await userEvent.keyboard('## Title{Shift>}{Enter}{Enter}{/Shift}- **bold** `code`');
		await settle();

		expect(editor.state.doc.toString()).toBe('## Title\n\n- **bold** `code`');
	});

	it('shows a mention link as a badge in the block being edited', async () => {
		const screen = render(ChatFormInput, { value: `see ${MENTION} ok` });
		const editor = editorIn(screen.container);

		editor.focus();
		editor.dispatch({ selection: { anchor: 0 } });
		await settle();

		const line = screen.container.querySelector('.cm-line');

		expect(line?.textContent).toContain('notes.md');
		expect(line?.textContent).not.toContain('file://');
		expect(editor.state.doc.toString()).toBe(`see ${MENTION} ok`);
	});

	it('steps over a mention badge as one unit', async () => {
		const screen = render(ChatFormInput, { value: `a ${MENTION}` });
		const editor = editorIn(screen.container);

		editor.focus();
		editor.dispatch({ selection: { anchor: editor.state.doc.length } });
		await settle();

		await userEvent.keyboard('{Backspace}');
		await settle();

		expect(editor.state.doc.toString()).toBe('a ');
	});

	it('renders every block like a chat message once blurred', async () => {
		const screen = render(ChatFormInput, { value: '# Title\n\nSome **bold** text' });
		const editor = editorIn(screen.container);

		editor.contentDOM.blur();
		await vi.waitFor(() => {
			expect(screen.container.querySelector('.cm-md-block h1')?.textContent).toBe('Title');
			expect(screen.container.querySelector('.cm-md-block strong')?.textContent).toBe('bold');
		});
	});

	it('lets the form handle a key first and keeps it from the editor', async () => {
		const onKeydown = vi.fn((event: KeyboardEvent) => {
			if (event.key === 'Enter') event.preventDefault();
		});
		const screen = render(ChatFormInput, { onKeydown, value: 'hello' });
		const editor = editorIn(screen.container);

		editor.focus();
		editor.dispatch({ selection: { anchor: 5 } });
		await userEvent.keyboard('{Enter}');
		await settle();

		expect(onKeydown).toHaveBeenCalled();
		expect(editor.state.doc.toString()).toBe('hello');
	});

	it('leaves a paste the form handles out of the editor', async () => {
		const onPaste = vi.fn((event: ClipboardEvent) => event.preventDefault());
		const screen = render(ChatFormInput, { onPaste, value: '' });
		const editor = editorIn(screen.container);
		const clipboardData = new DataTransfer();

		clipboardData.setData('text/plain', 'pasted');
		editor.focus();
		editor.contentDOM.dispatchEvent(
			new ClipboardEvent('paste', { bubbles: true, cancelable: true, clipboardData })
		);
		await settle();

		expect(onPaste).toHaveBeenCalled();
		expect(editor.state.doc.toString()).toBe('');
	});
});
