// Guards how the chat form tells why its microphone can't record: a page without HTTPS (a
// llama-server opened over plain HTTP on the network or over Tailscale) can't reach the
// microphone at all, the browser's own errors are named, and the button shows the reason
// while staying tappable, so a tap on a phone, which shows no tooltip, can explain it.

import ChatFormActionRecordWrapper from './components/ChatFormActionRecordWrapper.svelte';
import {
	describeMicrophoneError,
	microphoneUnavailableReason,
	VOICE_MICROPHONE_MESSAGE
} from '@ggml-org/llama-voice';
import { VOICE_MODE_MESSAGE } from '#lib/constants/index.ts';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

afterEach(() => {
	vi.unstubAllGlobals();
});

describe('microphoneUnavailableReason', () => {
	it('names HTTPS when the page is not a secure context', () => {
		vi.stubGlobal('isSecureContext', false);

		expect(microphoneUnavailableReason()).toBe(VOICE_MICROPHONE_MESSAGE.insecure);
	});

	it('names the browser when it cannot capture audio', () => {
		vi.stubGlobal('isSecureContext', true);
		vi.stubGlobal('AudioContext', undefined);

		expect(microphoneUnavailableReason()).toBe(VOICE_MICROPHONE_MESSAGE.unsupported);
	});

	it('has nothing to say when the page can record', () => {
		vi.stubGlobal('isSecureContext', true);

		expect(microphoneUnavailableReason()).toBeNull();
	});
});

describe('describeMicrophoneError', () => {
	it.each([
		['NotAllowedError', VOICE_MICROPHONE_MESSAGE.denied],
		['SecurityError', VOICE_MICROPHONE_MESSAGE.denied],
		['NotFoundError', VOICE_MICROPHONE_MESSAGE.notFound],
		['NotReadableError', VOICE_MICROPHONE_MESSAGE.busy]
	])('names a %s', (name, message) => {
		expect(describeMicrophoneError(new DOMException('', name))).toBe(message);
	});

	it("keeps an unknown error's own message", () => {
		expect(describeMicrophoneError(new Error('boom'))).toBe(
			`${VOICE_MICROPHONE_MESSAGE.failed} boom`
		);
	});
});

// the tooltip's trigger wraps the button and takes its name, so the button is the inner one
describe('the record button', () => {
	it('shows the reason and stays tappable when the microphone cannot record', async () => {
		const onMicClick = vi.fn();
		const screen = render(ChatFormActionRecordWrapper, {
			micWarning: VOICE_MICROPHONE_MESSAGE.insecure,
			onMicClick
		});
		const button = screen.getByRole('button', { name: /^Recording unavailable: / }).last();

		await expect.element(button).toBeEnabled();
		await button.click();

		expect(onMicClick).toHaveBeenCalledOnce();
	});

	it('starts recording when nothing is wrong', async () => {
		const screen = render(ChatFormActionRecordWrapper, {});

		await expect
			.element(screen.getByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage }).last())
			.toBeEnabled();
	});

	it('stays disabled when no model can take audio, whatever the microphone', async () => {
		const screen = render(ChatFormActionRecordWrapper, {
			micWarning: VOICE_MICROPHONE_MESSAGE.insecure,
			unavailableReason: VOICE_MODE_MESSAGE.unavailable
		});

		await expect
			.element(screen.getByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage }).last())
			.toBeDisabled();
	});
});
