import ActionIconHarness from './components/ActionIconHarness.svelte';
import { Copy } from '@lucide/svelte';
import { describe, expect, it } from 'vitest';
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

describe('ActionIcon disabled states and cursor styling', () => {
	it('renders disabled action icon with cursor-not-allowed and without cursor-pointer on desktop', async () => {
		await page.viewport(1280, 900);

		const { container } = render(ActionIconHarness, {
			disabled: true,
			icon: Copy,
			tooltip: 'Code incomplete'
		});
		const wrapper = container.querySelector('span');

		expect(wrapper).not.toBeNull();
		expect(wrapper?.classList.contains('cursor-not-allowed')).toBe(true);
		expect(wrapper?.classList.contains('cursor-pointer')).toBe(false);

		const button = container.querySelector('button');

		expect(button).not.toBeNull();
		expect(button?.disabled).toBe(true);
		expect(button?.className).toContain('disabled:opacity-40');
	});

	it('renders enabled action icon normally without cursor-not-allowed', async () => {
		await page.viewport(1280, 900);

		const { container } = render(ActionIconHarness, {
			disabled: false,
			icon: Copy,
			tooltip: 'Copy code'
		});
		const button = container.querySelector('button');

		expect(button).not.toBeNull();
		expect(button?.disabled).toBe(false);
		expect(button?.classList.contains('cursor-not-allowed')).toBe(false);
	});

	it('merges custom cursor class into Tooltip.Trigger replacing default cursor-pointer', () => {
		const { container } = render(ActionIconHarness, {
			triggerClass: 'cursor-not-allowed'
		});
		const trigger = container.firstElementChild as HTMLElement;

		expect(trigger).not.toBeNull();
		expect(trigger.classList.contains('cursor-not-allowed')).toBe(true);
		expect(trigger.classList.contains('cursor-pointer')).toBe(false);
	});

	it('renders CodeBlockActions disabled with not-allowed cursor and disabled button', async () => {
		await page.viewport(1280, 900);

		const { container } = render(ActionIconHarness, {
			code: 'echo "hello"',
			disabled: true,
			language: 'bash',
			testCodeBlockActions: true
		});
		const wrapper = container.querySelector('.code-block-actions span');

		expect(wrapper).not.toBeNull();
		expect(wrapper?.classList.contains('cursor-not-allowed')).toBe(true);
		expect(wrapper?.classList.contains('cursor-pointer')).toBe(false);

		const button = container.querySelector('button');

		expect(button).not.toBeNull();
		expect(button?.disabled).toBe(true);
		expect(button?.getAttribute('aria-label')).toBe('Code incomplete');
		expect(button?.className).toContain('disabled:opacity-40');
	});
});
