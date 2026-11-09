// Guards the height of the streaming reasoning box: page zoom shrinks the viewport in
// CSS px, so a fixed cap would cover most of a zoomed window (#30811).

import ChatMessageReasoningBlock from '$lib/components/app/chat/ChatMessages/ChatMessageReasoningBlock.svelte';
import { AgenticSectionType } from '$lib/enums';
import { afterEach, describe, expect, it } from 'vitest';
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const DESKTOP = { height: 900, width: 1280 };
// a 900px window at 150% zoom
const ZOOMED = { height: 600, width: 1280 };
const content = Array.from({ length: 200 }, (_, i) => `thought line ${i}`).join('\n');

function renderStreaming() {
	return render(ChatMessageReasoningBlock, {
		isStreaming: true,
		open: true,
		section: { content, type: AgenticSectionType.REASONING_PENDING }
	});
}

function boxHeight(): number {
	const box = document.querySelector('.reasoning-content') as HTMLElement;

	return box.getBoundingClientRect().height;
}

describe('streaming reasoning box height', () => {
	afterEach(async () => {
		await page.viewport(DESKTOP.width, DESKTOP.height);
	});

	it('stays within half of a zoomed viewport', async () => {
		await page.viewport(ZOOMED.width, ZOOMED.height);

		renderStreaming();

		await expect.poll(() => boxHeight()).toBeGreaterThan(0);
		expect(boxHeight()).toBeLessThanOrEqual(window.innerHeight / 2);
	});

	it('keeps the 28rem cap on a full-size window', async () => {
		await page.viewport(DESKTOP.width, DESKTOP.height);

		renderStreaming();

		await expect.poll(() => boxHeight()).toBeGreaterThan(0);
		expect(boxHeight()).toBe(448);
	});
});
