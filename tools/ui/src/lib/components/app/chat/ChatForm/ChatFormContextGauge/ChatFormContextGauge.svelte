<script lang="ts">
	import { untrack } from 'svelte';
	import * as HoverCard from '$lib/components/ui/hover-card';
	import { activeConversation, activeMessages } from '$lib/stores/conversations.svelte';
	import { chatStore, isChatStreaming, isLoading } from '$lib/stores/chat.svelte';
	import { formatParameters } from '$lib/utils/formatters';
	import { useContextGauge } from '$lib/hooks/use-context-gauge.svelte';
	import ContextGaugeDial from './ContextGaugeDial.svelte';
	import ContextGaugeDetails from './ContextGaugeDetails.svelte';
	import ContextGaugeLoadModel from './ContextGaugeLoadModel.svelte';
	import { colorLevelBgClass, colorLevelTextClass } from './context-gauge';

	const gauge = useContextGauge();

	$effect(() => {
		const conv = activeConversation();
		untrack(() => chatStore.setActiveProcessingConversation(conv?.id ?? null));
	});

	$effect(() => {
		const conv = activeConversation();
		const messages = activeMessages() as DatabaseMessage[];
		if (!conv) return;
		if (isLoading() || isChatStreaming()) return;

		if (messages.length === 0) {
			untrack(() => chatStore.clearProcessingState(conv.id));
			return;
		}

		untrack(() => chatStore.restoreProcessingStateFromMessages(messages, conv.id));
	});

	$effect(() => {
		gauge.startMonitoring();
	});

	const showProgressBar = $derived(
		gauge.contextTotal !== null &&
			gauge.contextTotal > 0 &&
			(gauge.activeModelId !== null || gauge.isActiveModelLoaded)
	);
</script>

<HoverCard.Root>
	<HoverCard.Trigger class="flex h-5 w-5 cursor-default items-center justify-center">
		<ContextGaugeDial percent={gauge.contextPercent} level={gauge.colorLevel} />
	</HoverCard.Trigger>

	<HoverCard.Content
		side="bottom"
		class="z-50 w-64 rounded-lg border border-border/50 bg-popover p-3 text-popover-foreground shadow-lg"
	>
		<div class="flex flex-col gap-2">
			<div class="flex items-center gap-2">
				<span class="font-medium">Context</span>
				<span class="text-muted-foreground">·</span>
				<span class="font-mono text-muted-foreground">
					{formatParameters(gauge.contextUsed)}
					/ {gauge.contextTotal !== null ? formatParameters(gauge.contextTotal) : '-'}
				</span>
			</div>

			{#if gauge.activeModelId !== null && !gauge.isActiveModelLoaded}
				<ContextGaugeLoadModel
					modelId={gauge.activeModelId}
					isLoading={gauge.isActiveModelLoading}
					onLoad={gauge.loadModel}
				/>
			{:else if showProgressBar}
				<div class="h-1.5 w-full overflow-hidden rounded-full bg-muted">
					<div
						class="h-full rounded-full transition-all duration-300 {colorLevelBgClass(
							gauge.colorLevel
						)}"
						style="width: {gauge.contextPercent}%"
					></div>
				</div>

				<div class="flex justify-between text-xs text-muted-foreground">
					<span>
						<span class={colorLevelTextClass(gauge.colorLevel)}>{gauge.contextPercent}%</span> used
					</span>
					<span>
						{formatParameters((gauge.contextTotal ?? 0) - gauge.contextUsed)} remaining
					</span>
				</div>
			{:else}
				<div class="text-xs text-muted-foreground">No context info available</div>
			{/if}

			{#if gauge.hasAnyUsage}
				<ContextGaugeDetails
					currentRead={gauge.currentRead}
					currentFresh={gauge.currentFresh}
					currentCache={gauge.currentCache}
					currentOutput={gauge.currentOutput}
					kvTotal={gauge.kvTotal}
					cumulativeRead={gauge.cumulativeRead}
					cumulativeOutput={gauge.cumulativeOutput}
					cumulativeCacheTotal={gauge.cumulativeCacheTotal}
					averageTokensPerSecond={gauge.averageTokensPerSecond}
					transientDetails={gauge.transientDetails}
				/>
			{/if}
		</div>
	</HoverCard.Content>
</HoverCard.Root>
