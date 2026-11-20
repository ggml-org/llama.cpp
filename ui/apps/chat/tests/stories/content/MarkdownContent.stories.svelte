<script lang="ts" module>
	import { AI_TUTORIAL_MD } from '../fixtures/ai-tutorial.js';
	import { MERMAID_DIAGRAMS_MD } from '../fixtures/mermaid-diagrams.js';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { MarkdownContent } from '#lib/components/app/index.ts';
	import { expect, waitFor } from 'storybook/test';

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
	}}
/>
