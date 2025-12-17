<script lang="ts">
	import { Clock, Gauge, WholeWord, BookOpenText, Sparkles } from '@lucide/svelte';
	import { BadgeChatStatistic } from '$lib/components/app';

	interface Props {
		predictedTokens: number;
		predictedMs: number;
		promptTokens?: number;
		promptMs?: number;
	}

	let { predictedTokens, predictedMs, promptTokens, promptMs }: Props = $props();

	let tokensPerSecond = $derived((predictedTokens / predictedMs) * 1000);
	let timeInSeconds = $derived((predictedMs / 1000).toFixed(2));

	let promptTokensPerSecond = $derived(
		promptTokens !== undefined && promptMs !== undefined
			? (promptTokens / promptMs) * 1000
			: undefined
	);
	let promptTimeInSeconds = $derived(
		promptMs !== undefined ? (promptMs / 1000).toFixed(2) : undefined
	);
	let hasPromptStats = $derived(
		promptTokens !== undefined &&
			promptMs !== undefined &&
			promptTokensPerSecond !== undefined &&
			promptTimeInSeconds !== undefined
	);
</script>

<div class="flex flex-col gap-1">
	{#if hasPromptStats}
		<div class="flex flex-wrap items-center gap-2">
			<span class="inline-flex items-center gap-1">
				<BookOpenText class="h-3.5 w-3.5" />
				<span>Reading:</span>
			</span>

			<BadgeChatStatistic icon={WholeWord} value="{promptTokens} tokens" />

			<BadgeChatStatistic icon={Clock} value="{promptTimeInSeconds}s" />

			<BadgeChatStatistic icon={Gauge} value="{promptTokensPerSecond!.toFixed(2)} tokens/s" />
		</div>
	{/if}

	<div class="flex flex-wrap items-center gap-2">
		<span class="inline-flex items-center gap-1">
			<Sparkles class="h-3.5 w-3.5" />
			<span>Generation:</span>
		</span>

		<BadgeChatStatistic icon={WholeWord} value="{predictedTokens} tokens" />

		<BadgeChatStatistic icon={Clock} value="{timeInSeconds}s" />

		<BadgeChatStatistic icon={Gauge} value="{tokensPerSecond.toFixed(2)} tokens/s" />
	</div>
</div>
