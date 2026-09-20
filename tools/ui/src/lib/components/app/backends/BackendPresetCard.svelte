<script lang="ts">
	import BackendPresetIcon from './BackendPresetIcon.svelte';
	import { Check } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import * as Card from '$lib/components/ui/card';
	import type { BackendPreset } from '$lib/types';

	interface Props {
		preset: BackendPreset;
		onClick?: () => void;
		selected?: boolean;
		dimmed?: boolean;
		/** A backend already points at this provider. */
		added?: boolean;
	}

	let { added = false, dimmed = false, onClick, preset, selected = false }: Props = $props();

	// an added provider is a label, not an action: no hover, no selection
	let interactive = $derived(!added && Boolean(onClick));
</script>

<Card.Root
	class={`relative gap-3! select-none bg-muted/30 p-4 transition-[background-color,opacity,transform] duration-150 ${interactive ? 'cursor-pointer hover:bg-muted/50 hover:opacity-100 active:scale-[0.99]' : ''} ${selected && !added ? 'bg-muted/30 ring-1 ring-primary/40' : ''} ${dimmed || added ? 'opacity-50' : ''}`}
	onclick={interactive ? onClick : undefined}
>
	{#if added}
		<Badge class="absolute top-2 right-2 gap-1 px-1.5 text-[10px]" variant="secondary">
			<Check class="h-3 w-3" />
			Added
		</Badge>
	{/if}

	<div class="flex min-w-0 items-center gap-2">
		<BackendPresetIcon class="h-5 w-5" {preset} />

		<h4 class="min-w-0 flex-1 truncate font-medium">{preset.name}</h4>
	</div>

	<p class="text-xs text-muted-foreground">{preset.description}</p>
</Card.Root>
