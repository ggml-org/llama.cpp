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
	play={async ({ canvas, userEvent }) => {
		const before = await canvas.findByRole('button', { name: 'before' });
		const after = await canvas.findByRole('button', { name: 'after' });
		const leftArrow = await canvas.findByRole('button', { name: 'Scroll left' });

		await waitFor(() => {
			expect(leftArrow).toBeDisabled();
		});

		before.focus();
		await userEvent.tab();

		await expect(after).toHaveFocus();
	}}
>
	<div>
		<button type="button">before</button>
		<HorizontalScrollCarousel class="w-96">
			<div class="h-12 w-12 shrink-0 bg-muted"></div>
			<div class="h-12 w-12 shrink-0 bg-muted"></div>
		</HorizontalScrollCarousel>
		<button type="button">after</button>
	</div>
</Story>

<Story
	asChild
	name="ArrowsInTabOrderWhenScrollable"
	play={async ({ canvas, userEvent }) => {
		const before = await canvas.findByRole('button', { name: 'before' });
		const rightArrow = await canvas.findByRole('button', { name: 'Scroll right' });

		await waitFor(() => {
			expect(rightArrow).not.toBeDisabled();
		});

		before.focus();
		await userEvent.tab();

		await expect(rightArrow).toHaveFocus();
	}}
>
	<div>
		<button type="button">before</button>
		<HorizontalScrollCarousel class="w-48">
			{#each [...Array(20).keys()] as i (i)}
				<div class="h-12 w-24 shrink-0 bg-muted">{i}</div>
			{/each}
		</HorizontalScrollCarousel>
	</div>
</Story>
