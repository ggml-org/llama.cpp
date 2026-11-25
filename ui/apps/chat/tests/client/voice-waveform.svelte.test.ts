// Guards VoiceWaveform's playback with real clicks, which a browser needs before it plays audio (the stories' clicks
// are scripted, so their play functions cover the rest): play and pause toggle, and one clip plays at a time.

import { demoClip } from '../stories/voice/fixtures';
import { VoiceWaveform } from '@ggml-org/llama-ui/voice';
import { describe, expect, it } from 'vitest';
import { page, userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

describe('VoiceWaveform playback', () => {
	it('toggles between play and pause', async () => {
		const clip = demoClip(3, 1);

		render(VoiceWaveform, { durationMs: clip.durationMs, peaks: clip.peaks, src: clip.src });

		await userEvent.click(page.getByRole('button', { name: 'Play voice message' }));
		await expect.element(page.getByRole('button', { name: 'Pause voice message' })).toBeVisible();

		await userEvent.click(page.getByRole('button', { name: 'Pause voice message' }));
		await expect.element(page.getByRole('button', { name: 'Play voice message' })).toBeVisible();
	});

	it('pauses one clip when another starts', async () => {
		const first = demoClip(3, 2);
		const second = demoClip(3, 3);

		render(VoiceWaveform, { label: 'First clip', peaks: first.peaks, src: first.src });
		render(VoiceWaveform, { label: 'Second clip', peaks: second.peaks, src: second.src });

		await userEvent.click(page.getByRole('button', { name: 'Play first clip' }));
		await expect.element(page.getByRole('button', { name: 'Pause first clip' })).toBeVisible();

		await userEvent.click(page.getByRole('button', { name: 'Play second clip' }));
		await expect.element(page.getByRole('button', { name: 'Pause second clip' })).toBeVisible();
		await expect.element(page.getByRole('button', { name: 'Play first clip' })).toBeVisible();

		await userEvent.click(page.getByRole('button', { name: 'Pause second clip' }));
	});
});
