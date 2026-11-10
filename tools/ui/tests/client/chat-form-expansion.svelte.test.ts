// Guards when the main chat form expands into its large editor and collapses back:
// on its own once a message asks for it, by hand, and back once it is cleared.

import ChatFormTestWrapper from './components/ChatFormTestWrapper.svelte';
import { EditorView } from '@codemirror/view';
import { conversationsStore } from '$lib/stores';
import { tick } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

async function renderForm(initialValue = '') {
	const { container } = render(ChatFormTestWrapper, { expandable: true, initialValue });

	await tick();

	const content = container.querySelector<HTMLElement>('.cm-content');
	const editor = content && EditorView.findFromDOM(content);

	if (!editor) throw new Error('editor not rendered');

	const write = async (text: string) => {
		editor.dispatch({ changes: { from: 0, insert: text, to: editor.state.doc.length } });
		await tick();
	};
	const isExpanded = () => !!container.querySelector('.chat-form-input.expanded');
	// on a phone-sized screen the sheet ignores the key while it moves
	const sheetMoves = () =>
		container
			.querySelector('[data-slot="input-area"]')!
			.getAnimations()
			.filter((animation) => !(animation instanceof CSSTransition)).length;
	const toggle = async () => {
		await expect.poll(sheetMoves).toBe(0);
		await userEvent.click(
			container.querySelector<HTMLElement>(
				`button[aria-label="${isExpanded() ? 'Collapse' : 'Expand'} editor"]`
			)!
		);
		await expect.poll(sheetMoves).toBe(0);
	};

	return { container, editor, isExpanded, toggle, write };
}

describe('ChatForm expansion', () => {
	afterEach(() => {
		conversationsStore.activeConversation = null;
	});

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

	it('leaves a restored draft in the small form until it is written in', async () => {
		const { isExpanded, write } = await renderForm('# Title\n\n- item\n');

		expect(isExpanded()).toBe(false);

		await write('# Title\n\n- item\n- another\n');
		expect(isExpanded()).toBe(true);
	});

	it('starts collapsed in another conversation', async () => {
		const { isExpanded, write } = await renderForm();

		await write('# Title\n');
		expect(isExpanded()).toBe(true);

		conversationsStore.activeConversation = { id: 'other' } as DatabaseConversation;
		await tick();

		expect(isExpanded()).toBe(false);
	});

	it('says how to send while expanded', async () => {
		const { container, toggle } = await renderForm();

		expect(container.textContent).not.toContain('Enter to send');

		await toggle();
		await tick();

		expect(container.textContent).toContain('Cmd/Ctrl+Enter to send');
	});

	it('scrolls a long message inside the editor', async () => {
		const draft = Array.from({ length: 200 }, (_, i) => `line ${i + 1}`).join('\n');
		const { container } = await renderForm(draft);
		const input = container.querySelector<HTMLElement>('.chat-form-input')!;
		const scroller = container.querySelector<HTMLElement>('.cm-scroller')!;

		// stands in for the stylesheet's --max-message-height, which the tests do not load
		input.style.maxHeight = '200px';

		await expect.poll(() => scroller.clientHeight).toBeLessThanOrEqual(200);
		expect(scroller.scrollHeight).toBeGreaterThan(scroller.clientHeight);
	});

	it('mirrors the height of the text onto the input', async () => {
		const { container, editor, write } = await renderForm();
		const input = container.querySelector<HTMLElement>('.chat-form-input')!;

		await write('one\ntwo');
		await expect.poll(() => input.style.height).toBe(`${editor.contentDOM.offsetHeight}px`);
		expect(editor.contentDOM.offsetHeight).toBeGreaterThan(40);
	});
});
