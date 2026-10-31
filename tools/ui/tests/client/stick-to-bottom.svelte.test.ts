// Guards the follow rules of StickToBottom: right after a press, growth of the
// content stays put (the reader expanded something), while growth of an element
// passed to followGrowthOf - the chat form expanding on the reader's click - is
// still followed, so the latest message stays in view above it.

import { createStickToBottom } from '$lib/hooks/use-stick-to-bottom.svelte';
import { afterEach, describe, expect, it } from 'vitest';

let cleanup: (() => void)[] = [];

afterEach(() => {
	cleanup.forEach((fn) => fn());
	cleanup = [];
});

function setup() {
	const container = document.createElement('div');
	const content = document.createElement('div');
	const form = document.createElement('div');

	container.style.cssText = 'height: 200px; overflow-y: auto;';
	content.style.height = '600px';
	form.style.height = '50px';
	container.append(content, form);
	document.body.append(container);

	const stick = createStickToBottom();

	stick.attach(container, content);
	stick.scrollToBottom();
	// as the container's scroll handler does, so a later move up reads as the reader's
	stick.handleScroll();
	cleanup.push(() => {
		stick.detach();
		container.remove();
	});

	return { container, content, form, stick };
}

function press(container: HTMLElement) {
	container.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
}

async function resizeObserved() {
	// ResizeObserver reports in the next rendering steps
	await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}

function distanceFromBottom(el: HTMLElement) {
	return el.scrollHeight - el.clientHeight - el.scrollTop;
}

describe('StickToBottom', () => {
	it('keeps the viewport when the content grows right after a press', async () => {
		const { container, content } = setup();

		press(container);
		content.style.height = '900px';
		await resizeObserved();

		expect(distanceFromBottom(container)).toBeGreaterThan(200);
	});

	it('follows a followed element growing right after a press', async () => {
		const { container, form, stick } = setup();

		cleanup.push(stick.followGrowthOf(form));
		await resizeObserved();

		press(container);
		form.style.height = '350px';
		await resizeObserved();

		expect(distanceFromBottom(container)).toBeLessThanOrEqual(1);
	});

	it('leaves a reader who scrolled away where they are', async () => {
		const { container, form, stick } = setup();

		cleanup.push(stick.followGrowthOf(form));
		container.scrollTop = 0;
		stick.handleScroll();
		form.style.height = '350px';
		await resizeObserved();

		expect(container.scrollTop).toBe(0);
	});
});
