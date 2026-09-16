<script lang="ts">
	import { Plus } from '@lucide/svelte';
	import {
		LOCAL_BACKEND_ID,
		MODELS_VIEW_FAVORITES,
		MODELS_VIEW_LOCAL,
		MODELS_VIEW_REMOTE
	} from '$lib/constants';
	import { backendsStore } from '$lib/stores';

	interface Props {
		activeId: string;
		onAdd: () => void;
		onSelect: (viewId: string) => void;
	}

	let { activeId, onAdd, onSelect }: Props = $props();

	const tabClass = (active: boolean) => [
		'inline-flex shrink-0 items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium whitespace-nowrap transition',
		active
			? 'bg-muted text-foreground'
			: 'text-muted-foreground hover:bg-muted/60 hover:text-foreground'
	];

	// the remote view is only offered while a remote backend exists
	const hasRemoteBackends = $derived(
		backendsStore.enabled.some((backend) => backend.id !== LOCAL_BACKEND_ID)
	);
</script>

<div class="flex items-center gap-1 overflow-x-auto px-2 py-2">
	<button
		class={tabClass(activeId === MODELS_VIEW_FAVORITES)}
		onclick={() => onSelect(MODELS_VIEW_FAVORITES)}
		type="button"
	>
		Favorites
	</button>

	<button
		class={tabClass(activeId === MODELS_VIEW_LOCAL)}
		onclick={() => onSelect(MODELS_VIEW_LOCAL)}
		type="button"
	>
		Local
	</button>

	{#if hasRemoteBackends}
		<button
			class={tabClass(activeId === MODELS_VIEW_REMOTE)}
			onclick={() => onSelect(MODELS_VIEW_REMOTE)}
			type="button"
		>
			Remote
		</button>
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
