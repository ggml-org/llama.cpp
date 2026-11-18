// The sidebar's footer is the third of four nested surfaces: the page, the sidebar on it, a
// group inset in the sidebar and a row in the group. This holds the rule app.css states for
// them, on what the browser actually renders: a row's corners are concentric with its
// group's, so its radius is the group's less the padding between their edges.

import '../../src/app.css';
import SidebarNavigationActions from '$lib/components/app/navigation/SidebarNavigation/SidebarNavigationActions.svelte';
import SidebarNavigationFooter from '$lib/components/app/navigation/SidebarNavigation/SidebarNavigationFooter.svelte';
import { SIDEBAR_ACTIONS_ITEMS, SIDEBAR_FOOTER_ITEMS } from '$lib/constants';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

function px(value: string): number {
	return parseFloat(value);
}

async function renderFooter() {
	const screen = render(SidebarNavigationFooter, {});
	const rows = () => [...screen.container.querySelectorAll<HTMLElement>('[data-slot="button"]')];

	await vi.waitFor(() => expect(rows()).toHaveLength(SIDEBAR_FOOTER_ITEMS.length));

	const group = rows()[0].parentElement?.parentElement;

	if (!group) throw new Error('footer group not rendered');

	return { group, rows: rows() };
}

describe('nested surfaces', () => {
	// the footer is a desktop's and a tablet's: a phone lists its items with the sidebar's actions
	beforeEach(async () => {
		await page.viewport(1280, 900);
	});

	it('round the footer group from its tokens', async () => {
		const { group } = await renderFooter();
		const root = getComputedStyle(document.documentElement);
		const style = getComputedStyle(group);

		// the tokens resolve, so the group is not left square or unpadded
		expect(px(style.borderTopLeftRadius)).toBeGreaterThan(0);
		expect(px(style.paddingTop)).toBeGreaterThan(0);
		expect(root.getPropertyValue('--surface-group-radius').trim()).not.toBe('');
		expect(root.getPropertyValue('--surface-group-padding').trim()).not.toBe('');
	});

	it('keep each footer row concentric with its group', async () => {
		const { group, rows } = await renderFooter();
		const style = getComputedStyle(group);
		const expected = px(style.borderTopLeftRadius) - px(style.paddingTop);

		for (const row of rows) {
			const radius = getComputedStyle(row).borderTopLeftRadius;

			expect(px(radius), row.textContent ?? '').toBeCloseTo(expected, 1);
		}
	});

	it('inset each footer row by the group padding the radius accounts for', async () => {
		const { group, rows } = await renderFooter();
		const style = getComputedStyle(group);
		const box = group.getBoundingClientRect();
		// the hairline border sits outside the padding the radius is reduced by
		const inner = box.left + px(style.borderLeftWidth) + px(style.paddingLeft);

		for (const row of rows) {
			expect(row.getBoundingClientRect().left).toBeCloseTo(inner, 0);
		}
	});

	it('round the top actions of the sidebar like the footer rows', async () => {
		const { rows: footerRows } = await renderFooter();
		const expected = getComputedStyle(footerRows[0]).borderTopLeftRadius;
		const top = render(SidebarNavigationActions, { class: '' });
		const rows = () => [...top.container.querySelectorAll<HTMLElement>('[data-slot="button"]')];

		await vi.waitFor(() => expect(rows()).toHaveLength(SIDEBAR_ACTIONS_ITEMS.length));

		for (const row of rows()) {
			expect(getComputedStyle(row).borderTopLeftRadius, row.textContent ?? '').toBe(expected);
		}
	});
});
