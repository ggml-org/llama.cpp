<script lang="ts">
	import { colorLevelBgClass, colorLevelTextClass } from './context-gauge';
	import ContextGaugeDetails from './ContextGaugeDetails.svelte';
	import ContextGaugeLoadModel from './ContextGaugeLoadModel.svelte';
	import { gaugePopupClose } from './gauge-popup.svelte';
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { FoldVertical, Loader2 } from '@lucide/svelte';
	import { useContextGauge } from '#lib/hooks/use-context-gauge.svelte.ts';
	import { agenticStore, chatStore, conversationsStore } from '#lib/stores/index.ts';
	import { formatParameters } from '#lib/utils/formatters.ts';

	interface Props {
		/** The drawer brings its own title, so the panel drops the inline one there. */
		inDrawer?: boolean;
	}

	let { inDrawer = false }: Props = $props();

	const gauge = useContextGauge();

	// The gauge hook wraps a processing state instance that only follows the
	// live stream while its own monitoring flag is set, so the panel instance
	// starts monitoring like the dial does.
	$effect(() => {
		gauge.startMonitoring();
	});

	const showProgressBar = $derived(
		gauge.contextTotal !== null &&
			gauge.contextTotal > 0 &&
			(gauge.activeModelId !== null || gauge.isActiveModelLoaded)
	);

	const showCompactButton = $derived(chatStore.canCompact(conversationsStore.activeMessages));

	const activeConvId = $derived(conversationsStore.activeConversation?.id ?? null);

	// A summary in flight keeps the panel open for the numbers, so the action
	// reports the run instead of inviting a second one.
	const isCompacting = $derived(activeConvId !== null && chatStore.isCompacting(activeConvId));

	const isCompactDisabled = $derived.by(() => {
		if (!activeConvId) return true;

		return chatStore.isChatLoading(activeConvId) || agenticStore.isRunning(activeConvId);
	});

	function handleCompact() {
		gaugePopupClose();
		chatStore.compactConversation();
	}
</script>

{#snippet usage()}
	<span class="font-mono text-muted-foreground">
		{formatParameters(gauge.contextUsed)}
		/ {gauge.contextTotal !== null ? formatParameters(gauge.contextTotal) : '-'}
	</span>
{/snippet}

{#snippet compactAction()}
	{#if showCompactButton}
		<Button
			class="my-1 mt-3 w-full gap-1.5"
			disabled={isCompactDisabled}
			onclick={handleCompact}
			size="sm"
			variant="secondary"
		>
			{#if isCompacting}
				<Loader2 class="size-icon-md animate-spin" />
				Compacting...
			{:else}
				<FoldVertical class="size-icon-md" />
				Compact conversation
			{/if}
		</Button>
	{/if}
{/snippet}

{#snippet body()}
	{#if gauge.activeModelId !== null && !gauge.isActiveModelLoaded}
		<ContextGaugeLoadModel
			isLoading={gauge.isActiveModelLoading}
			modelId={gauge.activeModelId}
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

		<div class="flex justify-between text-caption text-muted-foreground max-md:text-body-sm">
			<span>
				<span class={colorLevelTextClass(gauge.colorLevel)}>{gauge.contextPercent}%</span> used
			</span>

			<span>
				{formatParameters(gauge.contextAvailable ?? 0)} remaining
			</span>
		</div>
	{:else}
		<div class="text-caption text-muted-foreground">No context info available</div>
	{/if}

	{#if gauge.hasAnyUsage}
		<ContextGaugeDetails
			averageTokensPerSecond={gauge.averageTokensPerSecond}
			cumulativeCacheTotal={gauge.cumulativeCacheTotal}
			cumulativeOutput={gauge.cumulativeOutput}
			cumulativeRead={gauge.cumulativeRead}
			currentCache={gauge.currentCache}
			currentFresh={gauge.currentFresh}
			currentOutput={gauge.currentOutput}
			currentRead={gauge.currentRead}
			kvTotal={gauge.kvTotal}
			transientDetails={gauge.transientDetails}
		/>
	{/if}
{/snippet}

<div class={['flex flex-col gap-2', inDrawer ? 'px-4 pb-4 max-md:gap-3.5' : 'px-2 py-1.5']}>
	{#if inDrawer}
		{@render usage()}
	{:else}
		<div class="flex items-center gap-2">
			<span class="font-medium">Context</span>

			<span class="text-muted-foreground">·</span>

			{@render usage()}
		</div>
	{/if}

	{@render body()}

	{@render compactAction()}
</div>
