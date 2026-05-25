<script lang="ts">
	import { Clock, Gauge, WholeWord, BookOpenText, Sparkles, Wrench, Layers } from '@lucide/svelte';
	import { ChatMessageStatisticsBadge } from '$lib/components/app';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ChatMessageStatsView } from '$lib/enums';
	import type { ChatMessageAgenticTimings } from '$lib/types/chat';
	import { formatPerformanceTime } from '$lib/utils';
	import { MS_PER_SECOND, DEFAULT_PERFORMANCE_TIME } from '$lib/constants';
	import type { Component } from 'svelte';

	interface Props {
		predictedTokens?: number;
		predictedMs?: number;
		promptTokens?: number;
		promptMs?: number;
		isLive?: boolean;
		isProcessingPrompt?: boolean;
		initialView?: ChatMessageStatsView;
		agenticTimings?: ChatMessageAgenticTimings;
		onActiveViewChange?: (view: ChatMessageStatsView) => void;
		hideSummary?: boolean;
	}

	let {
		predictedTokens,
		predictedMs,
		promptTokens,
		promptMs,
		isLive = false,
		isProcessingPrompt = false,
		initialView = ChatMessageStatsView.GENERATION,
		agenticTimings,
		onActiveViewChange,
		hideSummary = false
	}: Props = $props();

	let activeView: ChatMessageStatsView = $derived(initialView);
	let hasAutoSwitchedToGeneration = $state(false);

	$effect(() => {
		onActiveViewChange?.(activeView);
	});

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

	let tokensPerSecond = $derived(
		hasGenerationStats ? (predictedTokens! / predictedMs!) * MS_PER_SECOND : 0
	);
	let formattedTime = $derived(
		predictedMs !== undefined ? formatPerformanceTime(predictedMs) : DEFAULT_PERFORMANCE_TIME
	);

	let promptTokensPerSecond = $derived(
		promptTokens !== undefined && promptMs !== undefined && promptMs > 0
			? (promptTokens / promptMs) * MS_PER_SECOND
			: undefined
	);

	let formattedPromptTime = $derived(
		promptMs !== undefined ? formatPerformanceTime(promptMs) : undefined
	);

	let hasPromptStats = $derived(
		promptTokens !== undefined &&
			promptMs !== undefined &&
			promptTokensPerSecond !== undefined &&
			formattedPromptTime !== undefined
	);

	// In live mode, generation tab is disabled until we have generation stats
	let isGenerationDisabled = $derived(isLive && !hasGenerationStats);

	let hasAgenticStats = $derived(agenticTimings !== undefined && agenticTimings.toolCallsCount > 0);

	let agenticToolsPerSecond = $derived(
		hasAgenticStats && agenticTimings!.toolsMs > 0
			? (agenticTimings!.toolCallsCount / agenticTimings!.toolsMs) * MS_PER_SECOND
			: 0
	);

	let formattedAgenticToolsTime = $derived(
		hasAgenticStats ? formatPerformanceTime(agenticTimings!.toolsMs) : DEFAULT_PERFORMANCE_TIME
	);

	let agenticTotalTimeMs = $derived(
		hasAgenticStats
			? agenticTimings!.toolsMs + agenticTimings!.llm.predicted_ms + agenticTimings!.llm.prompt_ms
			: 0
	);

	let formattedAgenticTotalTime = $derived(formatPerformanceTime(agenticTotalTimeMs));
</script>

{#snippet viewButton(opts: {
	view: ChatMessageStatsView;
	icon: Component;
	label: string;
	tooltipText: string;
	disabled?: boolean;
})}
	{@const IconComponent = opts.icon}
	<Tooltip.Root>
		<Tooltip.Trigger>
			<!-- prevent another nested button element -->
			{#snippet child({ props })}
				<button
					{...props}
					type="button"
					class="inline-flex h-5 w-5 items-center justify-center rounded-sm transition-colors {activeView ===
					opts.view
						? 'bg-background text-foreground shadow-sm'
						: opts.disabled
							? 'cursor-not-allowed opacity-40'
							: 'hover:text-foreground'}"
					onclick={() => !opts.disabled && (activeView = opts.view)}
					disabled={opts.disabled}
				>
					<IconComponent class="h-3 w-3" />

					<span class="sr-only">{opts.label}</span>
				</button>
			{/snippet}
		</Tooltip.Trigger>

		<Tooltip.Content>
			<p>{opts.tooltipText}</p>
		</Tooltip.Content>
	</Tooltip.Root>
{/snippet}

<div class="inline-flex items-center text-xs text-muted-foreground">
	<div class="inline-flex items-center rounded-sm bg-muted-foreground/15 p-0.5">
		{#if hasPromptStats || isLive}
			{@render viewButton({
				view: ChatMessageStatsView.READING,
				icon: BookOpenText,
				label: 'Reading',
				tooltipText: 'Reading (prompt processing)'
			})}
		{/if}

		{@render viewButton({
			view: ChatMessageStatsView.GENERATION,
			icon: Sparkles,
			label: 'Generation',
			tooltipText: isGenerationDisabled
				? 'Generation (waiting for tokens...)'
				: 'Generation (token output)',
			disabled: isGenerationDisabled
		})}

		{#if hasAgenticStats}
			{@render viewButton({
				view: ChatMessageStatsView.TOOLS,
				icon: Wrench,
				label: 'Tools',
				tooltipText: 'Tool calls'
			})}

			{#if !hideSummary}
				{@render viewButton({
					view: ChatMessageStatsView.SUMMARY,
					icon: Layers,
					label: 'Summary',
					tooltipText: 'Agentic summary'
				})}
			{/if}
		{/if}
	</div>

	<div class="flex items-center gap-1 px-2">
		{#if activeView === ChatMessageStatsView.GENERATION && hasGenerationStats}
			<ChatMessageStatisticsBadge
				class="bg-transparent"
				icon={WholeWord}
				value="{predictedTokens?.toLocaleString()} tokens"
				tooltipLabel="Generated tokens"
			/>

			<ChatMessageStatisticsBadge
				class="bg-transparent"
				icon={Clock}
				value={formattedTime}
				tooltipLabel="Generation time"
			/>

			<ChatMessageStatisticsBadge
				class="bg-transparent"
				icon={Gauge}
				value="{tokensPerSecond.toFixed(2)} t/s"
				tooltipLabel="Generation speed"
			/>
		{:else if activeView === ChatMessageStatsView.TOOLS && hasAgenticStats}
			<ChatMessageStatisticsBadge
				class="bg-transparent"
				icon={Wrench}
				value="{agenticTimings!.toolCallsCount} calls"
				tooltipLabel="Tool calls executed"
			/>

			<ChatMessageStatisticsBadge
				class="bg-transparent"
				icon={Clock}
				value={formattedAgenticToolsTime}
				tooltipLabel="Tool execution time"
			/>

			<ChatMessageStatisticsBadge
				class="bg-transparent"
				icon={Gauge}
				value="{agenticToolsPerSecond.toFixed(2)} calls/s"
				tooltipLabel="Tool execution rate"
			/>
		{:else if activeView === ChatMessageStatsView.SUMMARY && hasAgenticStats}
			<ChatMessageStatisticsBadge
				class="bg-transparent"
				icon={Layers}
				value="{agenticTimings!.turns} turns"
				tooltipLabel="Agentic turns (LLM calls)"
			/>

			<ChatMessageStatisticsBadge
				class="bg-transparent"
				icon={WholeWord}
				value="{agenticTimings!.llm.predicted_n.toLocaleString()} tokens"
				tooltipLabel="Total tokens generated"
			/>

			<ChatMessageStatisticsBadge
				class="bg-transparent"
				icon={Clock}
				value={formattedAgenticTotalTime}
				tooltipLabel="Total time (LLM + tools)"
			/>
		{:else if hasPromptStats}
			<ChatMessageStatisticsBadge
				class="bg-transparent"
				icon={WholeWord}
				value="{promptTokens} tokens"
				tooltipLabel="Prompt tokens"
			/>

			<ChatMessageStatisticsBadge
				class="bg-transparent"
				icon={Clock}
				value={formattedPromptTime ?? '0s'}
				tooltipLabel="Prompt processing time"
			/>

			<ChatMessageStatisticsBadge
				class="bg-transparent"
				icon={Gauge}
				value="{promptTokensPerSecond!.toFixed(2)} tokens/s"
				tooltipLabel="Prompt processing speed"
			/>
		{/if}
	</div>
</div>
