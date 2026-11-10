// Guards when the main chat form expands into its large editor and collapses back:
// on its own once a message asks for it, by hand, and back once it is cleared.

import ChatFormTestWrapper from './components/ChatFormTestWrapper.svelte';
import { EditorView } from '@codemirror/view';
import { tick } from 'svelte';
import { describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

async function renderForm() {
	const { container } = render(ChatFormTestWrapper, { expandable: true });

	await tick();

	const content = container.querySelector<HTMLElement>('.cm-content');
	const editor = content && EditorView.findFromDOM(content);

	if (!editor) throw new Error('editor not rendered');

	const write = async (text: string) => {
		editor.dispatch({ changes: { from: 0, insert: text, to: editor.state.doc.length } });
		await tick();
	};
	const isExpanded = () => !!container.querySelector('.chat-form-input.expanded');
	const toggle = () =>
		userEvent.click(
			container.querySelector<HTMLElement>(
				`button[aria-label="${isExpanded() ? 'Collapse' : 'Expand'} editor"]`
			)!
		);

	return { container, editor, isExpanded, toggle, write };
}

describe('ChatForm expansion', () => {
	it('stays small for a line that only starts like markdown', async () => {
		const { isExpanded, write } = await renderForm();

		for (const text of ['*', '-5 degrees', '# Title', '1.5x faster']) {
			await write(text);
			expect(isExpanded(), text).toBe(false);
		}
	});

	it('expands on a finished markdown line or a fourth line', async () => {
		const { isExpanded, write } = await renderForm();

		await write('# Title\n');
		expect(isExpanded()).toBe(true);

		await write('');
		expect(isExpanded()).toBe(false);

		await write('one\ntwo\nthree\nfour');
		expect(isExpanded()).toBe(true);
	});

	it('stays expanded while the message has text', async () => {
		const { isExpanded, write } = await renderForm();

		await write('one\ntwo\nthree\nfour');
		await write('one');

		expect(isExpanded()).toBe(true);
	});

	it('keeps a collapse by hand until the message is cleared', async () => {
		const { isExpanded, toggle, write } = await renderForm();

		await write('# Title\n');
		await toggle();
		await tick();
		expect(isExpanded()).toBe(false);

		await write('# Title\n\n- item\n');
		expect(isExpanded()).toBe(false);

		await write('');
		await write('# Title\n');
		expect(isExpanded()).toBe(true);
	});

	it('says how to send while expanded', async () => {
		const { container, toggle } = await renderForm();

		expect(container.textContent).not.toContain('Enter to send');

		await toggle();
		await tick();

		expect(container.textContent).toContain('Cmd/Ctrl+Enter to send');
	});

	it('mirrors the height of the text onto the input', async () => {
		const { container, editor, write } = await renderForm();
		const input = container.querySelector<HTMLElement>('.chat-form-input')!;

		await write('one\ntwo');
		await expect.poll(() => input.style.height).toBe(`${editor.contentDOM.offsetHeight}px`);
		expect(editor.contentDOM.offsetHeight).toBeGreaterThan(40);
	});
});
