<script lang="ts">
	import { Loader2, Plus } from '@lucide/svelte';
	import { backendsModelsStore } from '$lib/stores';
	import type { Backend } from '$lib/types';

	interface Props {
		activeId: string;
		backends: Backend[];
		onAdd: () => void;
		onSelect: (backendId: string) => void;
	}

	let { activeId, backends, onAdd, onSelect }: Props = $props();
</script>

<div class="flex items-center gap-1 overflow-x-auto border-b border-border/50 px-2 py-2">
	{#each backends as backend (backend.id)}
		{@const state = backendsModelsStore.get(backend.id)}
		<button
			class={[
				'inline-flex shrink-0 items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium whitespace-nowrap transition',
				activeId === backend.id
					? 'bg-muted text-foreground'
					: 'text-muted-foreground hover:bg-muted/60 hover:text-foreground'
			]}
			onclick={() => onSelect(backend.id)}
			type="button"
		>
			{backend.name}

			{#if state.loading}
				<Loader2 class="h-3 w-3 animate-spin" />
			{:else if state.error}
				<span class="h-1.5 w-1.5 rounded-full bg-destructive" title={state.error}></span>
			{/if}
		</button>
	{/each}

	{#if backends.length === 0}
		<span class="px-1 text-xs text-muted-foreground whitespace-nowrap">No backends configured</span>
	{/if}

	<button
		aria-label="Add backend"
		class="ml-auto shrink-0 rounded-full p-1 text-muted-foreground transition hover:bg-muted/60 hover:text-foreground"
		onclick={onAdd}
		type="button"
	>
		<Plus class="h-3.5 w-3.5" />
	</button>
</div>
