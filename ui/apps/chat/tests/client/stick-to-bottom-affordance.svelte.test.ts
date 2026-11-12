// Guards when StickToBottom offers the way back down: not for a small step up from the
// bottom, which still reads as near it, only once the reader is well above it, and
// never once they are back at the bottom.

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

	container.style.cssText = 'height: 200px; overflow-y: auto;';
	content.style.height = '1000px';
	container.append(content);
	document.body.append(container);

	const stick = createStickToBottom();

	stick.attach(container, content);
	stick.scrollToBottom();
	stick.handleScroll();
	cleanup.push(() => {
		stick.detach();
		container.remove();
	});

	return { container, stick };
}

function scrollTo(
	container: HTMLElement,
	stick: ReturnType<typeof createStickToBottom>,
	top: number
) {
	container.scrollTop = top;
	stick.handleScroll();
}

describe('StickToBottom affordance', () => {
	it('is not offered a small step above the bottom', () => {
		const { container, stick } = setup();

		scrollTo(container, stick, container.scrollTop - 100);

		expect(stick.atBottom).toBe(false);
		expect(stick.farFromBottom).toBe(false);
	});

	it('is offered well above the bottom and withdrawn back at it', () => {
		const { container, stick } = setup();
		const bottom = container.scrollTop;

		scrollTo(container, stick, 0);
		expect(stick.farFromBottom).toBe(true);

		scrollTo(container, stick, bottom);
		expect(stick.farFromBottom).toBe(false);
		expect(stick.atBottom).toBe(true);
	});
});
