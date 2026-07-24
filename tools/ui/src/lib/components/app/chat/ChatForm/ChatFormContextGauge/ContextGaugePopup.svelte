<script lang="ts">
	import { formatParameters } from '$lib/utils/formatters';
	import { useContextGauge } from '$lib/hooks/use-context-gauge.svelte';
	import ContextGaugeDetails from './ContextGaugeDetails.svelte';
	import ContextGaugeLoadModel from './ContextGaugeLoadModel.svelte';
	import { colorLevelBgClass, colorLevelTextClass } from './context-gauge';
	import {
		gaugePopup,
		gaugeCardEnter,
		gaugeCardLeave,
		gaugePopupClose
	} from '$lib/stores/context-gauge-popup.svelte';

	const gauge = useContextGauge();

	let cardEl = $state<HTMLElement | null>(null);

	// Any press outside the card and outside the dial closes the card.
	// Presses on the dial are excluded because the dial handles its own
	// toggle; the listener only exists while the card is open.
	$effect(() => {
		if (!gaugePopup.open) return;

		const onPointerDown = (event: PointerEvent) => {
			const target = event.target;
			if (!(target instanceof Node)) return;
			if (cardEl?.contains(target)) return;
			if (target instanceof Element && target.closest('[data-context-gauge-trigger]')) return;
			gaugePopupClose();
		};

		document.addEventListener('pointerdown', onPointerDown, true);
		return () => document.removeEventListener('pointerdown', onPointerDown, true);
	});

	const showProgressBar = $derived(
		gauge.contextTotal !== null &&
			gauge.contextTotal > 0 &&
			(gauge.activeModelId !== null || gauge.isActiveModelLoaded)
	);
</script>

{#if gaugePopup.open}
	<div
		role="status"
		bind:this={cardEl}
		class="absolute z-50 w-64 -translate-x-1/2 rounded-lg border border-border/50 bg-popover p-3 text-popover-foreground shadow-lg"
		style="left: {gaugePopup.centerX}px; bottom: {gaugePopup.bottom}px"
		onpointerenter={gaugeCardEnter}
		onpointerleave={gaugeCardLeave}
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
	</div>
{/if}
