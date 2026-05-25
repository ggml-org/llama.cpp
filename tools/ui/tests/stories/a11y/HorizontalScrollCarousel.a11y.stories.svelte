<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import HorizontalScrollCarousel from '$lib/components/app/misc/HorizontalScrollCarousel.svelte';
	import { expect, waitFor } from 'storybook/test';

	const { Story } = defineMeta({
		title: 'Components/HorizontalScrollCarousel/Accessibility',
		component: HorizontalScrollCarousel,
		parameters: {
			layout: 'centered'
		},
		tags: ['!dev']
	});
</script>

<Story
	asChild
	name="ArrowsNotInTabOrderWhenNotScrollable"
	play={async ({ canvas }) => {
		const leftArrow = await canvas.findByRole('button', { name: 'Scroll left' });
		const rightArrow = await canvas.findByRole('button', { name: 'Scroll right' });

		await waitFor(() => {
			expect(leftArrow).toBeDisabled();
			expect(rightArrow).toBeDisabled();
		});
	}}
>
	<HorizontalScrollCarousel class="w-96">
		<div class="h-12 w-12 shrink-0 bg-muted"></div>
		<div class="h-12 w-12 shrink-0 bg-muted"></div>
	</HorizontalScrollCarousel>
</Story>

<Story
	asChild
	name="ArrowsInTabOrderWhenScrollable"
	play={async ({ canvas }) => {
		const rightArrow = await canvas.findByRole('button', { name: 'Scroll right' });

		await waitFor(() => {
			expect(rightArrow).not.toBeDisabled();
		});
	}}
>
	<HorizontalScrollCarousel class="w-48">
		{#each [...Array(20).keys()] as i (i)}
			<div class="h-12 w-24 shrink-0 bg-muted">{i}</div>
		{/each}
	</HorizontalScrollCarousel>
</Story>
