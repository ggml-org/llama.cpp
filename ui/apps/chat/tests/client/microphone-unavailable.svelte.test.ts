// Guards how the chat form tells why its microphone can't record: a page without HTTPS (a
// llama-server opened over plain HTTP on the network or over Tailscale) can't reach the
// microphone at all, the browser's own errors are named, and the button shows the reason
// while staying tappable, so a tap on a phone, which shows no tooltip, can explain it.

import ChatFormActionRecordWrapper from './components/ChatFormActionRecordWrapper.svelte';
import { MICROPHONE_UNAVAILABLE_MESSAGE } from '$lib/constants';
import { describeMicrophoneError, microphoneUnavailableReason } from '$lib/utils/browser-only';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

afterEach(() => {
	vi.unstubAllGlobals();
});

describe('microphoneUnavailableReason', () => {
	it('names HTTPS when the page is not a secure context', () => {
		vi.stubGlobal('isSecureContext', false);

		expect(microphoneUnavailableReason()).toBe(MICROPHONE_UNAVAILABLE_MESSAGE.insecure);
	});

	it('names the browser when it has no recorder', () => {
		vi.stubGlobal('isSecureContext', true);
		vi.stubGlobal('MediaRecorder', undefined);

		expect(microphoneUnavailableReason()).toBe(MICROPHONE_UNAVAILABLE_MESSAGE.unsupported);
	});

	it('has nothing to say when the page can record', () => {
		vi.stubGlobal('isSecureContext', true);

		expect(microphoneUnavailableReason()).toBeNull();
	});
});

describe('describeMicrophoneError', () => {
	it.each([
		['NotAllowedError', MICROPHONE_UNAVAILABLE_MESSAGE.denied],
		['SecurityError', MICROPHONE_UNAVAILABLE_MESSAGE.denied],
		['NotFoundError', MICROPHONE_UNAVAILABLE_MESSAGE.notFound],
		['NotReadableError', MICROPHONE_UNAVAILABLE_MESSAGE.busy]
	])('names a %s', (name, message) => {
		expect(describeMicrophoneError(new DOMException('', name))).toBe(message);
	});

	it("keeps an unknown error's own message", () => {
		expect(describeMicrophoneError(new Error('boom'))).toBe(
			`${MICROPHONE_UNAVAILABLE_MESSAGE.failed} boom`
		);
	});
});

// the tooltip's trigger wraps the button and takes its name, so the button is the inner one
describe('the record button', () => {
	it('shows the reason and stays tappable when the microphone cannot record', async () => {
		const onMicClick = vi.fn();
		const screen = render(ChatFormActionRecordWrapper, {
			micWarning: MICROPHONE_UNAVAILABLE_MESSAGE.insecure,
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
			.element(screen.getByRole('button', { name: 'Start recording' }).last())
			.toBeEnabled();
	});

	it('stays disabled for a model without audio, whatever the microphone', async () => {
		const screen = render(ChatFormActionRecordWrapper, {
			hasAudioModality: false,
			micWarning: MICROPHONE_UNAVAILABLE_MESSAGE.insecure
		});

		await expect
			.element(screen.getByRole('button', { name: 'Start recording' }).last())
			.toBeDisabled();
	});
});
