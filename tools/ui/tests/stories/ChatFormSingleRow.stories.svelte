<script lang="ts" module>
	import Preview from './fixtures/ChatFormSingleRowPreview.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Scrapbook/ChatForm single row'
	});

	const SHORT = 'What is the meaning of life?';
	const TWO_LINES = 'What is the meaning of life?\nAnswer in one sentence.';

	// the default is --chat-form-single-row-radius in app.css; the card is about 3.375rem tall
	const RADII = [
		['2rem', 'a pill (2rem)'],
		['1.5rem', '1.5rem, like the two-row form on a desktop'],
		['1.25rem', '1.25rem (the default)'],
		['1rem', '1rem'],
		['0.75rem', '0.75rem']
	] as const;
</script>

<Story name="Empty">
	<Preview />
</Story>

<Story name="Short text">
	<Preview initial={SHORT} />
</Story>

<Story name="Two lines">
	<Preview initial={TWO_LINES} />
</Story>

<Story name="Before: two rows">
	<Preview today />
</Story>

<Story name="Radius variations">
	<div class="flex flex-col gap-6">
		{#each RADII as [radius, label] (radius)}
			<div class="flex flex-col gap-1.5">
				<span class="text-xs text-muted-foreground">{label}</span>

				<Preview initial={SHORT} {radius} />
			</div>
		{/each}
	</div>
</Story>
