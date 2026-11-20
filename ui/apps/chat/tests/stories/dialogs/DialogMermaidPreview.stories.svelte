<script lang="ts" module>
	import { FLOWCHART_SVG } from '../fixtures/mermaid-preview.js';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { DialogMermaidPreview } from '#lib/components/app/index.ts';
	import { expect, waitFor } from 'storybook/test';

	// the preview a diagram block's preview button opens; it fills the window on every device
	const { Story } = defineMeta({
		component: DialogMermaidPreview,
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Llama Chat/Dialogs/DialogMermaidPreview'
	});
</script>

<script lang="ts">
	let open = $state(true);
</script>

<Story
	asChild
	name="Default"
	play={async () => {
		const content = await waitFor(() => {
			const el = document.querySelector<HTMLElement>('[data-slot="dialog-content"]');

			expect(el?.querySelector('.mermaid-preview')).not.toBeNull();

			return el!;
		});
		const rect = content.getBoundingClientRect();

		await expect(rect.left).toBe(0);
		await expect(rect.top).toBe(0);
		await expect(rect.width).toBeCloseTo(window.innerWidth, 0);
		await expect(rect.height).toBeCloseTo(window.innerHeight, 0);
	}}
>
	<DialogMermaidPreview bind:open svgHtml={FLOWCHART_SVG} />
</Story>
