<script lang="ts">
	import { modelSizeLabel } from '../utils';
	import { Play, Square, X } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		isCustomized: boolean;
		isLoaded: boolean;
		onClose: () => void;
		onToggleLoad: () => void;
		onUseInNewChat: () => void;
		option: ModelOption;
	}

	let { isCustomized, isLoaded, onClose, onToggleLoad, onUseInNewChat, option }: Props = $props();

	let size = $derived(modelSizeLabel(option));
</script>

<div class="flex items-start gap-3 px-4 pt-4">
	<span
		class="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-muted text-xs font-medium text-muted-foreground"
	>
		{option.parsedId?.params ?? '—'}
	</span>

	<div class="min-w-0 flex-1">
		<div class="flex items-center gap-2">
			<span class="truncate text-base font-medium">{option.name}</span>

			{#if isCustomized}
				<Badge class="h-5 px-1.5 text-[10px]" variant="secondary">CUSTOMIZED</Badge>
			{/if}
		</div>

		<p class="truncate text-xs text-muted-foreground">
			{option.parsedId?.orgName ?? 'unknown'} · {isLoaded ? 'loaded' : 'not loaded'}{size
				? ` · ${size}`
				: ''}
		</p>
	</div>

	<Button aria-label="Close details" class="h-7 w-7" onclick={onClose} size="icon" variant="ghost">
		<X class="h-4 w-4" />
	</Button>
</div>

<div class="flex gap-2 px-4 pt-3">
	<Button class="flex-1 gap-1.5" onclick={onUseInNewChat} variant="outline">
		<Play class="h-3.5 w-3.5" />

		Use in New Chat
	</Button>

	<Button class="flex-1 gap-1.5" onclick={onToggleLoad} variant="outline">
		<Square class="h-3.5 w-3.5" />

		{isLoaded ? 'Eject Model' : 'Load Model'}
	</Button>
</div>
