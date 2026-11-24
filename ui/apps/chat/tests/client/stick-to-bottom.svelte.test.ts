// Guards the follow rules of StickToBottom: right after a press, growth of the
// content stays put (the reader expanded something), while growth of an element
// passed to followSizeOf - the chat form expanding on the reader's click - is
// still followed, so the latest message stays in view above it. A send glides to
// the bottom: eased, never jumping, and handed back the moment the reader scrolls.

import { createStickToBottom } from '#lib/hooks/use-stick-to-bottom.svelte.ts';
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

function frame() {
	return new Promise((resolve) => requestAnimationFrame(resolve));
}

/**
 * Waits until the container rests at its bottom. A glide counts animated
 * time, not wall time, so a runner that drops frames takes longer to get
 * there: the deadline is only a guard against a glide that never arrives,
 * set far beyond what one can take.
 */
const SETTLE_DEADLINE_MS = 20_000;

async function settled(container: HTMLElement) {
	const deadline = performance.now() + SETTLE_DEADLINE_MS;

	while (distanceFromBottom(container) > 1 && performance.now() < deadline) {
		await frame();
	}
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

		cleanup.push(stick.followSizeOf(form));
		await resizeObserved();

		press(container);
		form.style.height = '350px';
		await resizeObserved();

		expect(distanceFromBottom(container)).toBeLessThanOrEqual(1);
	});

	it('leaves a reader who scrolled away where they are', async () => {
		const { container, form, stick } = setup();

		cleanup.push(stick.followSizeOf(form));
		container.scrollTop = 0;
		stick.handleScroll();
		form.style.height = '350px';
		await resizeObserved();

		expect(container.scrollTop).toBe(0);
	});

	it('glides to the new bottom after a send instead of jumping there', async () => {
		const { container, content, stick } = setup();

		stick.glideToBottom();
		content.style.height = '1400px';

		const tops: number[] = [];

		// the first frames after the growth are on their way, not there yet
		await frame();
		tops.push(container.scrollTop);
		await frame();
		tops.push(container.scrollTop);

		expect(distanceFromBottom(container)).toBeGreaterThan(200);

		await settled(container);

		for (let i = 0; i < 5; i++) {
			tops.push(container.scrollTop);
			await frame();
		}

		expect(distanceFromBottom(container)).toBeLessThanOrEqual(1);
		// only ever down: no overshoot, no step back
		expect(tops).toEqual([...tops].sort((a, b) => a - b));
	});

	it('glides to layout that settles shortly after a send', async () => {
		const { container, content, stick } = setup();

		stick.glideToBottom();
		content.style.height = '1400px';
		await settled(container);

		content.style.height = '1700px';
		await resizeObserved();

		expect(distanceFromBottom(container)).toBeGreaterThan(100);

		await settled(container);

		expect(distanceFromBottom(container)).toBeLessThanOrEqual(1);
	});

	it('hands the page back the moment the reader scrolls', async () => {
		const { container, content, stick } = setup();

		stick.glideToBottom();
		content.style.height = '1400px';
		await frame();
		await frame();
		window.dispatchEvent(new WheelEvent('wheel'));

		const top = container.scrollTop;

		await resizeObserved();
		await frame();

		expect(container.scrollTop).toBe(top);
		expect(distanceFromBottom(container)).toBeGreaterThan(200);
	});

	it('keeps gliding while the reader types the next message', async () => {
		const { container, content, stick } = setup();
		const field = document.createElement('textarea');

		document.body.append(field);
		cleanup.push(() => field.remove());

		stick.glideToBottom();
		content.style.height = '1400px';
		await frame();
		field.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: ' ' }));
		await settled(container);

		expect(distanceFromBottom(container)).toBeLessThanOrEqual(1);
	});
});
