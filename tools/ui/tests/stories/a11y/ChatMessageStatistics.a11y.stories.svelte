<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatMessageStatistics from '$lib/components/app/chat/ChatMessages/ChatMessageStatistics/ChatMessageStatistics.svelte';
	import { expect } from 'storybook/test';

	const { Story } = defineMeta({
		title: 'Components/ChatMessageStatistics/Accessibility',
		component: ChatMessageStatistics,
		parameters: {
			layout: 'centered'
		},
		tags: ['!dev']
	});
</script>

<Story
	name="ViewButtonsSingleTabStop"
	args={{
		promptTokens: 100,
		promptMs: 500,
		predictedTokens: 200,
		predictedMs: 1000,
		agenticTimings: {
			turns: 1,
			toolCallsCount: 1,
			toolsMs: 500,
			llm: { predicted_n: 200, predicted_ms: 1000, prompt_n: 100, prompt_ms: 500 }
		},
		hideSummary: false,
		isLive: false
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
