<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ScrollCarousel } from '$lib/components/app';
	import { ScrollCarouselVariant } from '$lib/enums';
	import { expect, waitFor } from 'storybook/test';

	const { Story } = defineMeta({
		component: ScrollCarousel,
		parameters: {
			layout: 'centered'
		},
		tags: ['!dev'],
		title: 'Components/ScrollCarousel/Accessibility'
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

		<ScrollCarousel class="w-96" variant={ScrollCarouselVariant.CENTER}>
			<div class="h-12 w-12 shrink-0 bg-muted"></div>

			<div class="h-12 w-12 shrink-0 bg-muted"></div>
		</ScrollCarousel>

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

		<ScrollCarousel class="w-48" variant={ScrollCarouselVariant.CENTER}>
			{#each [...Array(20).keys()] as i (i)}
				<div class="h-12 w-24 shrink-0 bg-muted">{i}</div>
			{/each}
		</ScrollCarousel>
	</div>
</Story>
