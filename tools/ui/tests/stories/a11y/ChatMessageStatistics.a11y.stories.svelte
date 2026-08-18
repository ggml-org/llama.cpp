<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatMessageStatistics from '$lib/components/app/chat/ChatMessages/ChatMessageStatistics/ChatMessageStatistics.svelte';
	import { expect } from 'storybook/test';

	const { Story } = defineMeta({
		component: ChatMessageStatistics,
		parameters: {
			layout: 'centered'
		},
		tags: ['!dev'],
		title: 'Components/ChatMessageStatistics/Accessibility'
	});
</script>

<Story
	name="ViewButtonsSingleTabStop"
	args={{
		agenticTimings: {
			llm: { predicted_ms: 1000, predicted_n: 200, prompt_ms: 500, prompt_n: 100 },
			toolCallsCount: 1,
			toolsMs: 500,
			turns: 1
		},
		hideSummary: false,
		isLive: false,
		predictedMs: 1000,
		predictedTokens: 200,
		promptMs: 500,
		promptTokens: 100
	}}
	play={async ({ canvas, userEvent }) => {
		const reading = await canvas.findByRole('button', { name: 'Reading' });
		const generation = await canvas.findByRole('button', { name: 'Generation' });
		const tools = await canvas.findByRole('button', { name: 'Tools' });
		const summary = await canvas.findByRole('button', { name: 'Summary' });

		reading.focus();
		await expect(reading).toHaveFocus();

		await userEvent.tab();
		await expect(generation).toHaveFocus();

		await userEvent.tab();
		await expect(tools).toHaveFocus();

		await userEvent.tab();
		await expect(summary).toHaveFocus();
	}}
/>
