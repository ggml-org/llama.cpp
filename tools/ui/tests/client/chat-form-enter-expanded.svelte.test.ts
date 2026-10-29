// Guards the Enter-key contract of the expanded chat form: it is a writing
// surface, so plain Enter adds a line (continuing lists like any editor) and
// only Cmd/Ctrl+Enter submits the message.

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

async function renderExpanded() {
	const onSubmit = vi.fn();
	const { container } = render(ChatFormTestWrapper, { expandable: true, onSubmit });

	await tick();
	const toggle = [...container.querySelectorAll('button')].find((button) =>
		button.textContent?.includes('Expand editor')
	);

	if (!toggle) throw new Error('expand toggle not rendered');

	await userEvent.click(toggle);
	await tick();

	const editor = editorIn(container);

	await userEvent.click(editor.contentDOM);

	return { editor, onSubmit };
}

describe('ChatForm Enter in the expanded editor', () => {
	beforeEach(() => {
		settingsStore.updateConfig(SETTINGS_KEYS.SEND_ON_ENTER, true);
	});

	it('adds a line on Enter instead of submitting', async () => {
		const { editor, onSubmit } = await renderExpanded();

		await userEvent.keyboard('hello{Enter}world');
		await tick();

		expect(onSubmit).not.toHaveBeenCalled();
		expect(editor.state.doc.toString()).toBe('hello\nworld');
	});

	it('continues a list on Enter', async () => {
		const { editor, onSubmit } = await renderExpanded();

		await userEvent.keyboard('- one{Enter}two');
		await tick();

		expect(onSubmit).not.toHaveBeenCalled();
		expect(editor.state.doc.toString()).toBe('- one\n- two');
	});

	it.each(['Meta', 'Control'])('submits on %s+Enter', async (modifier) => {
		const { editor, onSubmit } = await renderExpanded();

		await userEvent.keyboard(`hello{${modifier}>}{Enter}{/${modifier}}`);
		await tick();

		expect(onSubmit).toHaveBeenCalledOnce();
		expect(editor.state.doc.toString()).toBe('hello');
	});
});
