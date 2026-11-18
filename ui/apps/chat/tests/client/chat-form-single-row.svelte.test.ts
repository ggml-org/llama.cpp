// Guards when the main chat form keeps a one-line message on the actions' row, and that
// a second line, or a line that wraps, takes the two rows without flipping back.

import ChatFormTestWrapper from './components/ChatFormTestWrapper.svelte';
import { EditorView } from '@codemirror/view';
import { tick } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const PHONE = { height: 844, width: 390 };
const DESKTOP = { height: 900, width: 1280 };

async function renderForm() {
	const { container } = render(ChatFormTestWrapper, { expandable: true, singleRow: true });

	await tick();

	const content = container.querySelector<HTMLElement>('.cm-content');
	const editor = content && EditorView.findFromDOM(content);

	if (!editor) throw new Error('editor not rendered');

	const card = container.querySelector<HTMLElement>('[data-slot="input-area"]')!;
	const isSingleRow = () => card.hasAttribute('data-single-row');
	const write = async (text: string) => {
		editor.dispatch({ changes: { from: 0, insert: text, to: editor.state.doc.length } });
		await tick();
		// the text's height arrives with the next layout
		await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
	};

	return { isSingleRow, write };
}

describe('ChatForm single row', () => {
	afterEach(async () => {
		await page.viewport(DESKTOP.width, DESKTOP.height);
	});

	it('keeps one line on the actions row, and a second line below them', async () => {
		await page.viewport(DESKTOP.width, DESKTOP.height);

		const { isSingleRow, write } = await renderForm();

		expect(isSingleRow()).toBe(true);

		await write('What is the meaning of life?');
		expect(isSingleRow()).toBe(true);

		await write('What is the meaning of life?\nIn one sentence.');
		expect(isSingleRow()).toBe(false);

		await write('What is the meaning of life?');
		expect(isSingleRow()).toBe(true);
	});

	it('leaves a line that wrapped on two rows until it is shorter than where it wrapped', async () => {
		await page.viewport(DESKTOP.width, DESKTOP.height);

		const { isSingleRow, write } = await renderForm();
		const long = 'a long question '.repeat(40).trim();

		await write(long);
		expect(isSingleRow()).toBe(false);

		await write(long.slice(0, -1));
		expect(isSingleRow()).toBe(false);

		await write('short');
		expect(isSingleRow()).toBe(true);
	});

	it('keeps the two rows on a phone', async () => {
		await page.viewport(PHONE.width, PHONE.height);

		const { isSingleRow } = await renderForm();

		expect(isSingleRow()).toBe(false);
	});
});
