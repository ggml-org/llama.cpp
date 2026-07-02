<script lang="ts">
	import * as HoverCard from '$lib/components/ui/hover-card';
	import { activeProcessingState } from '$lib/stores/chat.svelte';
	import { selectedModelContextSize } from '$lib/stores/models.svelte';
	import { formatParameters } from '$lib/utils/formatters';

	let contextTotal = $derived(
		selectedModelContextSize() ?? 4096
	);

	let contextUsed = $derived(
		activeProcessingState()?.contextUsed ?? 0
	);

	let contextPercent = $derived(
		contextTotal > 0 ? Math.round((contextUsed / contextTotal) * 100) : 0
	);

	let contextLabelColor = $derived.by(() => {
		if (contextPercent >= 90) return 'text-red-400';
		if (contextPercent >= 75) return 'text-amber-400';
		return 'text-green-400';
	});

	const CIRCUMFERENCE = 2 * Math.PI * 11;
</script>

<HoverCard.Root>
	<HoverCard.Trigger class="flex h-7 w-7 cursor-default items-center justify-center">
		<svg viewBox="0 0 32 32" fill="none" class="h-7 w-7">
			<!-- Background track -->
			<circle
				cx="16" cy="16" r="11"
				stroke="currentColor" stroke-opacity="0.1" stroke-width="2"
			/>
			<!-- Progress arc -->
			<circle
				cx="16" cy="16" r="11"
				class="transition-colors duration-300"
				class:text-green-400={contextPercent < 75}
				class:text-amber-400={contextPercent >= 75 && contextPercent < 90}
				class:text-red-400={contextPercent >= 90}
				stroke="currentColor"
				stroke-width="2"
				stroke-linecap="round"
				stroke-dasharray="{CIRCUMFERENCE}"
				stroke-dashoffset={CIRCUMFERENCE * (1 - contextPercent / 100)}
				transform="rotate(-90 16 16)"
			/>
		</svg>
	</HoverCard.Trigger>

	<HoverCard.Content side="bottom" class="z-50 w-64 rounded-lg border border-border/50 bg-popover p-3 text-popover-foreground shadow-lg">
		<div class="flex flex-col gap-2">
			<div class="flex items-center gap-2">
				<span class="font-medium">Context</span>
				<span class="text-muted-foreground">·</span>
				<span class="font-mono text-muted-foreground">
					{formatParameters(contextUsed)}
					/ {formatParameters(contextTotal)}
				</span>
			</div>

			<div class="h-1.5 w-full overflow-hidden rounded-full bg-muted">
				<div
					class="h-full rounded-full transition-all duration-300"
					class:bg-green-500={contextPercent < 75}
					class:bg-amber-500={contextPercent >= 75 && contextPercent < 90}
					class:bg-red-500={contextPercent >= 90}
					style="width: {contextPercent}%"
				></div>
			</div>

			<div class="flex justify-between text-xs text-muted-foreground">
				<span>
					<span class={contextLabelColor}>{contextPercent}%</span> used
				</span>
				<span>
					{formatParameters(contextTotal - contextUsed)} remaining
				</span>
			</div>
		</div>
	</HoverCard.Content>
</HoverCard.Root>
