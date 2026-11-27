<script lang="ts" module>
	import { AI_TUTORIAL_MD } from '../fixtures/ai-tutorial.js';
	import { MARKDOWN_TABLES_MD } from '../fixtures/markdown-tables.js';
	import { MERMAID_DIAGRAMS_MD } from '../fixtures/mermaid-diagrams.js';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { MarkdownContent } from '#lib/components/app/index.ts';
	import { expect, userEvent, waitFor } from 'storybook/test';

	const { Story } = defineMeta({
		component: MarkdownContent,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Content/MarkdownContent'
	});
</script>

<Story
	args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]', content: AI_TUTORIAL_MD }}
	name="Default"
/>

<Story
	args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]', content: MARKDOWN_TABLES_MD }}
	name="Tables"
	play={async ({ canvasElement }) => {
		const tables = await waitFor(() => {
			const found = [...canvasElement.querySelectorAll('table')];

			expect(found).toHaveLength(5);

			return found;
		});
		const numbers = tables[1];
		const sizeCell = numbers.querySelectorAll('td')[2];
		const sizeHeader = numbers.querySelectorAll('th')[2];
		const wrapper = numbers.closest('.table-wrapper') as HTMLElement;

		// the lean look: no frame, no filled header, one hairline under each row
		await expect(getComputedStyle(wrapper).borderTopWidth).toBe('0px');
		await expect(getComputedStyle(sizeHeader).backgroundColor).toBe('rgba(0, 0, 0, 0)');
		await expect(getComputedStyle(sizeHeader).borderBottomWidth).toBe('1px');
		await expect(getComputedStyle(sizeCell).borderLeftWidth).toBe('0px');

		// a right-aligned column stays aligned in the header and in the cells
		await expect(getComputedStyle(sizeHeader).textAlign).toBe('end');
		await expect(getComputedStyle(sizeCell).textAlign).toBe('end');

		// markup models put in cells still renders: a list, and a wrapped line
		await expect(canvasElement.querySelector('td ul li')).not.toBeNull();
		await expect(canvasElement.querySelector('td br')).not.toBeNull();

		// too many columns for the column: the wrapper scrolls, the page does not
		const wide = (tables[4] as HTMLElement).closest('.table-wrapper') as HTMLElement;

		await expect(wide.scrollWidth).toBeGreaterThan(wide.clientWidth);
		await expect(document.documentElement.scrollWidth).toBeLessThanOrEqual(window.innerWidth);
		// and it scrolls on a width a model name sits on, rather than crumbling
		await expect(wide.querySelector('td')!.getBoundingClientRect().width).toBeGreaterThanOrEqual(
			100
		);
	}}
/>

<Story
	args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]', content: MERMAID_DIAGRAMS_MD }}
	name="Diagrams"
	play={async ({ canvasElement }) => {
		// MarkdownContent renders in batches, so the broken last block can trail the others
		await waitFor(
			() => {
				expect(canvasElement.querySelectorAll('pre.mermaid svg.pele')).toHaveLength(4);
				expect(canvasElement.querySelector('pre.mermaid .mermaid-error')).not.toBeNull();
			},
			{ timeout: 5000 }
		);

		const error = canvasElement.querySelector('pre.mermaid .mermaid-error');

		await expect(error?.textContent).toContain('line 2');
		await expect(error?.closest('pre')).toBeVisible();

		// `click ... href` must not turn a node into a live link, as under Mermaid's strict mode
		const flowchart = canvasElement.querySelector('pre.mermaid svg.pele');

		await expect(flowchart?.textContent).toContain('llama-server');
		await expect(canvasElement.querySelector('pre.mermaid a')).toBeNull();
		await expect(
			canvasElement.querySelector('pre.mermaid [href], pre.mermaid [*|href]')
		).toBeNull();

		// The preview button opens the diagram in a dialog that fills the window
		await userEvent.click(canvasElement.querySelector<HTMLElement>('.preview-code-btn')!);

		const preview = await waitFor(() => {
			const el = document.querySelector<HTMLElement>('.mermaid-preview');

			expect(el).not.toBeNull();

			return el!;
		});

		await expect(preview.getBoundingClientRect().width).toBeCloseTo(window.innerWidth, 0);
		await expect(preview.getBoundingClientRect().height).toBeCloseTo(window.innerHeight, 0);

		await userEvent.keyboard('{Escape}');
		await waitFor(() => expect(document.querySelector('.mermaid-preview')).toBeNull());
	}}
/>
