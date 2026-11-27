// Guards the Escape contract of the chat form: while a reply streams Escape stops it, as the
// stop button does, unless something else takes the key first - an open menu or drawer, a
// picker, a selection in the editor, the system message editor or a message edit. A tooltip
// does not take it.

import ChatFormTestWrapper from './components/ChatFormTestWrapper.svelte';
import { EditorView } from '@codemirror/view';
import { OPEN_OVERLAY_SELECTOR } from '#lib/constants/index.ts';
import { chatStore } from '#lib/stores/index.ts';
import { tick } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const PHONE = { height: 844, width: 390 };
const DESKTOP = { height: 900, width: 1280 };
const ADD_MENU_TRIGGER_SELECTOR = '.file-upload-button';
const TOOLTIP_SELECTOR = '[data-slot="tooltip-content"]';
// a tooltip opens after its hover delay
const TOOLTIP_TIMEOUT_MS = 5000;
const MENTION_PICKER_TEXT = 'File search is unavailable';

async function renderForm({ isLoading = true, systemMode = false } = {}) {
	const onStop = vi.fn();
	const { container } = render(ChatFormTestWrapper, { isLoading, onStop, systemMode });

	await tick();

	const content = container.querySelector<HTMLElement>('.cm-content');
	const editor = content && EditorView.findFromDOM(content);

	if (!editor) throw new Error('editor not rendered');

	await userEvent.click(editor.contentDOM);

	return { container, editor, onStop };
}

describe('ChatForm Escape', () => {
	afterEach(async () => {
		chatStore.clearEditMode();
		await page.viewport(DESKTOP.width, DESKTOP.height);
	});

	it('stops the reply from the editor', async () => {
		const { onStop } = await renderForm();

		await userEvent.keyboard('{Escape}');

		expect(onStop).toHaveBeenCalledOnce();
	});

	it('stops the reply with the focus outside the form', async () => {
		const { editor, onStop } = await renderForm();

		editor.contentDOM.blur();
		await userEvent.keyboard('{Escape}');

		expect(onStop).toHaveBeenCalledOnce();
	});

	it('does nothing with no reply streaming', async () => {
		const { onStop } = await renderForm({ isLoading: false });

		await userEvent.keyboard('{Escape}');

		expect(onStop).not.toHaveBeenCalled();
	});

	// the add menu is a dropdown on a desktop and a drawer on a phone
	it.each([
		{ overlay: 'menu', viewport: DESKTOP },
		{ overlay: 'drawer', viewport: PHONE }
	])('closes the open add $overlay first', async ({ viewport }) => {
		await page.viewport(viewport.width, viewport.height);

		const { container, onStop } = await renderForm();
		const overlay = () => document.querySelector(OPEN_OVERLAY_SELECTOR);

		await userEvent.click(container.querySelector<HTMLElement>(ADD_MENU_TRIGGER_SELECTOR)!);
		await expect.poll(overlay).toBeTruthy();

		await userEvent.keyboard('{Escape}');
		await expect.poll(overlay).toBeNull();
		expect(onStop).not.toHaveBeenCalled();

		await userEvent.keyboard('{Escape}');
		expect(onStop).toHaveBeenCalledOnce();
	});

	it('stops the reply with a tooltip showing', async () => {
		const { container, onStop } = await renderForm();

		await userEvent.hover(container.querySelector<HTMLElement>(ADD_MENU_TRIGGER_SELECTOR)!);
		await expect
			.poll(() => document.querySelector(TOOLTIP_SELECTOR), { timeout: TOOLTIP_TIMEOUT_MS })
			.toBeTruthy();

		await userEvent.keyboard('{Escape}');
		expect(onStop).toHaveBeenCalledOnce();
	});

	it('closes the mention picker first', async () => {
		const { onStop } = await renderForm();

		await userEvent.keyboard('@m');
		await expect.poll(() => document.body.textContent).toContain(MENTION_PICKER_TEXT);

		await userEvent.keyboard('{Escape}');
		await expect.poll(() => document.body.textContent).not.toContain(MENTION_PICKER_TEXT);
		expect(onStop).not.toHaveBeenCalled();
	});

	it('drops a selection in the editor first', async () => {
		const { editor, onStop } = await renderForm();

		await userEvent.keyboard('hello');
		editor.dispatch({ selection: { anchor: 0, head: editor.state.doc.length } });

		await userEvent.keyboard('{Escape}');
		expect(editor.state.selection.main.empty).toBe(true);
		expect(onStop).not.toHaveBeenCalled();

		await userEvent.keyboard('{Escape}');
		expect(onStop).toHaveBeenCalledOnce();
	});

	it('leaves the key to the system message editor', async () => {
		const { onStop } = await renderForm({ systemMode: true });

		await userEvent.keyboard('{Escape}');

		expect(onStop).not.toHaveBeenCalled();
	});

	it('leaves the key to a message edit', async () => {
		const { onStop } = await renderForm();

		chatStore.setEditModeActive(() => {});
		await userEvent.keyboard('{Escape}');

		expect(onStop).not.toHaveBeenCalled();
	});
});
