// Guards the Enter-key contract of the chat form against the
// fenced-code-block flow: while the caret sits inside a fenced
// block region - closed, or still OPEN while the user is typing
// one - plain Enter adds a line instead of submitting the message.
// The form gates submit; the editor then adds the line itself.

import ChatFormTestWrapper from './components/ChatFormTestWrapper.svelte';
import { EditorView } from '@codemirror/view';
import { SETTINGS_KEYS } from '$lib/constants';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { tick } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

function editorIn(container: HTMLElement): EditorView {
	const el = container.querySelector<HTMLElement>('.cm-content');
	const view = el && EditorView.findFromDOM(el);

	if (!view) throw new Error('editor not rendered');

	return view;
}

describe('ChatForm Enter in code blocks', () => {
	beforeEach(() => {
		settingsStore.updateConfig(SETTINGS_KEYS.SEND_ON_ENTER, true);
	});

	it('adds a line after a still-open fence instead of submitting', async () => {
		const onSubmit = vi.fn();
		const { container } = render(ChatFormTestWrapper, { onSubmit });

		await tick();

		const editor = editorIn(container);

		await userEvent.click(editor.contentDOM);
		await userEvent.keyboard('```');
		await tick();

		await userEvent.keyboard('{Enter}');
		await tick();

		expect(onSubmit).not.toHaveBeenCalled();
		expect(editor.state.doc.toString()).toBe('```\n');
	});

	it('keeps adding lines while the block stays open', async () => {
		const onSubmit = vi.fn();
		const { container } = render(ChatFormTestWrapper, { onSubmit });

		await tick();

		const editor = editorIn(container);

		await userEvent.click(editor.contentDOM);
		await userEvent.keyboard('```js');
		await tick();

		await userEvent.keyboard('{Enter}');
		await userEvent.keyboard('const a = 1;');
		await userEvent.keyboard('{Enter}');
		await tick();

		expect(onSubmit).not.toHaveBeenCalled();
		expect(editor.state.doc.toString()).toBe('```js\nconst a = 1;\n');
	});

	it('submits when the caret is before the opening fence', async () => {
		const onSubmit = vi.fn();
		const { container } = render(ChatFormTestWrapper, { onSubmit });

		await tick();

		const editor = editorIn(container);

		await userEvent.click(editor.contentDOM);
		await userEvent.keyboard('```');
		await tick();

		editor.dispatch({ selection: { anchor: 0 } });

		await userEvent.keyboard('{Enter}');
		await tick();

		expect(onSubmit).toHaveBeenCalledTimes(1);
	});

	it('submits on Enter outside a code block', async () => {
		const onSubmit = vi.fn();
		const { container } = render(ChatFormTestWrapper, { onSubmit });

		await tick();

		const editor = editorIn(container);

		await userEvent.click(editor.contentDOM);
		await userEvent.keyboard('hello');
		await tick();

		await userEvent.keyboard('{Enter}');
		await tick();

		expect(onSubmit).toHaveBeenCalledTimes(1);
	});

	it('submits on Ctrl+Enter even inside a code block', async () => {
		const onSubmit = vi.fn();
		const { container } = render(ChatFormTestWrapper, { onSubmit });

		await tick();

		const editor = editorIn(container);

		await userEvent.click(editor.contentDOM);
		await userEvent.keyboard('```');
		await tick();

		await userEvent.keyboard('{Control>}{Enter}{/Control}');
		await tick();

		expect(onSubmit).toHaveBeenCalledTimes(1);
	});
});
