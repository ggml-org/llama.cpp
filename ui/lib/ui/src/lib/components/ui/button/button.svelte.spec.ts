import { createRawSnippet } from 'svelte';
import { describe, expect, it, vi } from 'vite-plus/test';
import { render } from 'vitest-browser-svelte';
import Button from './button.svelte';

function textSnippet(text: string) {
	return createRawSnippet(() => ({
		render: () => `<span>${text}</span>`,
		setup: () => {}
	}));
}

describe('Button', () => {
	it('renders children and handles clicks', async () => {
		const onclick = vi.fn();
		const screen = await render(Button, {
			props: { onclick, children: textSnippet('click me') }
		});

		await expect.element(screen.getByRole('button', { name: 'click me' })).toBeInTheDocument();

		await screen.getByRole('button', { name: 'click me' }).click();

		expect(onclick).toHaveBeenCalledOnce();
	});

	it('applies variant classes', async () => {
		const screen = await render(Button, {
			props: { variant: 'destructive', children: textSnippet('danger') }
		});
		const el = screen.getByRole('button', { name: 'danger' }).element() as HTMLElement;

		expect(el.dataset.variant).toBe('destructive');
		expect(el.className).toContain('bg-(--button-danger-background)');
	});

	it('renders an anchor when href is set', async () => {
		const screen = await render(Button, {
			props: { href: 'https://example.com', children: textSnippet('link') }
		});

		expect(screen.getByRole('link', { name: 'link' })).toBeInTheDocument();
	});

	it('disables the button when disabled is set', async () => {
		const screen = await render(Button, {
			props: { disabled: true, children: textSnippet('nope') }
		});

		expect(screen.getByRole('button', { name: 'nope' })).toBeDisabled();
	});
});
