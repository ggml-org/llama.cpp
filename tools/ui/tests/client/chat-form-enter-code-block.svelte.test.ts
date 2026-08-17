// Guards the Enter-key contract of the chat form against the
// fenced-code-block flow: while the caret sits inside a fenced
// block region - closed, or still OPEN while the user is typing
// one - plain Enter adds a line instead of submitting the message.
// The textarea path is covered here end-to-end (the ChatFormInputRich
// consumes the same case locally; see chat-form-input-rich).

import ChatFormTestWrapper from './components/ChatFormTestWrapper.svelte';
import { SETTINGS_KEYS } from '$lib/constants';
import { settingsStore } from '$lib/stores/settings.svelte';
import { tick } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

function textareaIn(container: HTMLElement): HTMLTextAreaElement {
	const el = container.querySelector('textarea');

	if (!(el instanceof HTMLTextAreaElement)) throw new Error('textarea not rendered');

	return el;
}

describe('ChatForm Enter in code blocks', () => {
	beforeEach(() => {
		settingsStore.updateConfig(SETTINGS_KEYS.SEND_ON_ENTER, true);
	});

	it('adds a line after a still-open fence instead of submitting', async () => {
		const onSubmit = vi.fn();
		const { container } = render(ChatFormTestWrapper, { onSubmit });

		await tick();

		const textarea = textareaIn(container);

		await userEvent.click(textarea);
		await userEvent.keyboard('```');
		await tick();

		await userEvent.keyboard('{Enter}');
		await tick();

		expect(onSubmit).not.toHaveBeenCalled();
		expect(textarea.value).toBe('```\n');
	});

	it('keeps adding lines while the block stays open', async () => {
		const onSubmit = vi.fn();
		const { container } = render(ChatFormTestWrapper, { onSubmit });

		await tick();

		const textarea = textareaIn(container);

		await userEvent.click(textarea);
		await userEvent.keyboard('```js');
		await tick();

		await userEvent.keyboard('{Enter}');
		await userEvent.keyboard('const a = 1;');
		await userEvent.keyboard('{Enter}');
		await tick();

		expect(onSubmit).not.toHaveBeenCalled();
		expect(textarea.value).toBe('```js\nconst a = 1;\n');
	});

	it('submits when the caret is before the opening fence', async () => {
		const onSubmit = vi.fn();
		const { container } = render(ChatFormTestWrapper, { onSubmit });

		await tick();

		const textarea = textareaIn(container);

		await userEvent.click(textarea);
		await userEvent.keyboard('```');
		await tick();

		textarea.setSelectionRange(0, 0);

		await userEvent.keyboard('{Enter}');
		await tick();

		expect(onSubmit).toHaveBeenCalledTimes(1);
	});

	it('submits on Enter outside a code block', async () => {
		const onSubmit = vi.fn();
		const { container } = render(ChatFormTestWrapper, { onSubmit });

		await tick();

		const textarea = textareaIn(container);

		await userEvent.click(textarea);
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

		const textarea = textareaIn(container);

		await userEvent.click(textarea);
		await userEvent.keyboard('```');
		await tick();

		await userEvent.keyboard('{Control>}{Enter}{/Control}');
		await tick();

		expect(onSubmit).toHaveBeenCalledTimes(1);
	});
});
