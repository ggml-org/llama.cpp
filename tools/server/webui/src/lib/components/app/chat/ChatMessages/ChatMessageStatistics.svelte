<script lang="ts">
	import { Clock, Gauge, WholeWord, BookOpenText, Sparkles } from '@lucide/svelte';
	import { BadgeChatStatistic } from '$lib/components/app';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ChatMessageStatsView } from '$lib/enums';

	interface Props {
		predictedTokens?: number;
		predictedMs?: number;
		promptTokens?: number;
		promptMs?: number;
		// Live mode: when true, shows stats during streaming
		isLive?: boolean;
		// Whether prompt processing is still in progress
		isProcessingPrompt?: boolean;
		// Initial view to show (defaults to READING in live mode)
		initialView?: ChatMessageStatsView;
	}

	let {
		predictedTokens,
		predictedMs,
		promptTokens,
		promptMs,
		isLive = false,
		isProcessingPrompt = false,
		initialView = ChatMessageStatsView.GENERATION
	}: Props = $props();

	let activeView: ChatMessageStatsView = $state(initialView);
	let hasAutoSwitchedToGeneration = $state(false);

	// In live mode: auto-switch to GENERATION tab when prompt processing completes
	$effect(() => {
		if (isLive) {
			// Auto-switch to generation tab only when prompt processing is done (once)
			if (
				!hasAutoSwitchedToGeneration &&
				!isProcessingPrompt &&
				predictedTokens &&
				predictedTokens > 0
			) {
				activeView = ChatMessageStatsView.GENERATION;
				hasAutoSwitchedToGeneration = true;
			} else if (!hasAutoSwitchedToGeneration) {
				// Stay on READING while prompt is still being processed
				activeView = ChatMessageStatsView.READING;
			}
		}
	});

	let hasGenerationStats = $derived(
		predictedTokens !== undefined &&
			predictedTokens > 0 &&
			predictedMs !== undefined &&
			predictedMs > 0
	);

	let tokensPerSecond = $derived(hasGenerationStats ? (predictedTokens! / predictedMs!) * 1000 : 0);
	let timeInSeconds = $derived(
		predictedMs !== undefined ? (predictedMs / 1000).toFixed(2) : '0.00'
	);

	let promptTokensPerSecond = $derived(
		promptTokens !== undefined && promptMs !== undefined && promptMs > 0
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

	// In live mode, generation tab is disabled until we have generation stats
	let isGenerationDisabled = $derived(isLive && !hasGenerationStats);
</script>

<div class="inline-flex items-center text-xs text-muted-foreground">
	<div class="inline-flex items-center rounded-sm bg-muted-foreground/15 p-0.5">
		{#if hasPromptStats || isLive}
			<Tooltip.Root>
				<Tooltip.Trigger>
					<button
						type="button"
						class="inline-flex h-5 w-5 items-center justify-center rounded-sm transition-colors {activeView ===
						ChatMessageStatsView.READING
							? 'bg-background text-foreground shadow-sm'
							: 'hover:text-foreground'}"
						onclick={() => (activeView = ChatMessageStatsView.READING)}
					>
						<BookOpenText class="h-3 w-3" />
						<span class="sr-only">Reading</span>
					</button>
				</Tooltip.Trigger>
				<Tooltip.Content>
					<p>Reading (prompt processing)</p>
				</Tooltip.Content>
			</Tooltip.Root>
		{/if}
		<Tooltip.Root>
			<Tooltip.Trigger>
				<button
					type="button"
					class="inline-flex h-5 w-5 items-center justify-center rounded-sm transition-colors {activeView ===
					ChatMessageStatsView.GENERATION
						? 'bg-background text-foreground shadow-sm'
						: isGenerationDisabled
							? 'cursor-not-allowed opacity-40'
							: 'hover:text-foreground'}"
					onclick={() => !isGenerationDisabled && (activeView = ChatMessageStatsView.GENERATION)}
					disabled={isGenerationDisabled}
				>
					<Sparkles class="h-3 w-3" />
					<span class="sr-only">Generation</span>
				</button>
			</Tooltip.Trigger>
			<Tooltip.Content>
				<p>
					{isGenerationDisabled
						? 'Generation (waiting for tokens...)'
						: 'Generation (token output)'}
				</p>
			</Tooltip.Content>
		</Tooltip.Root>
	</div>

	<div class="flex items-center gap-1 px-2">
		{#if activeView === ChatMessageStatsView.GENERATION && hasGenerationStats}
			<BadgeChatStatistic
				class="bg-transparent"
				icon={WholeWord}
				value="{predictedTokens?.toLocaleString()} tokens"
				tooltipLabel="Generated tokens"
			/>
			<BadgeChatStatistic
				class="bg-transparent"
				icon={Clock}
				value="{timeInSeconds}s"
				tooltipLabel="Generation time"
			/>
			<BadgeChatStatistic
				class="bg-transparent"
				icon={Gauge}
				value="{tokensPerSecond.toFixed(2)} tokens/s"
				tooltipLabel="Generation speed"
			/>
		{:else if hasPromptStats}
			<BadgeChatStatistic
				class="bg-transparent"
				icon={WholeWord}
				value="{promptTokens} tokens"
				tooltipLabel="Prompt tokens"
			/>
			<BadgeChatStatistic
				class="bg-transparent"
				icon={Clock}
				value="{promptTimeInSeconds}s"
				tooltipLabel="Prompt processing time"
			/>
			<BadgeChatStatistic
				class="bg-transparent"
				icon={Gauge}
				value="{promptTokensPerSecond!.toFixed(2)} tokens/s"
				tooltipLabel="Prompt processing speed"
			/>
		{/if}
	</div>
</div>
