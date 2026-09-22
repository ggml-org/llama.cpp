<script lang="ts">
	import { Boxes } from '@lucide/svelte';
	import { Logo } from '$lib/components/app';
	import { BackendIcon } from '$lib/components/app/backends';
	import { getBackend } from '$lib/utils/api-base';

	interface Props {
		/** Null renders the aggregate entry, which lists every provider's models. */
		backendId: string | null;
		count: number;
		isActive: boolean;
		isLocal: boolean;
		label: string;
		onSelect: () => void;
	}

	let { backendId, count, isActive, isLocal, label, onSelect }: Props = $props();
</script>

<button
	aria-pressed={isActive}
	class={[
		'flex w-full cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition',
		isActive ? 'bg-accent text-accent-foreground' : 'hover:bg-muted/50'
	]}
	onclick={onSelect}
	type="button"
>
	{#if isLocal}
		<Logo class="shrink-0" style="--size: 1rem" />
	{:else if backendId}
		<BackendIcon backend={getBackend(backendId)} class="h-4 w-4" />
	{:else}
		<Boxes class="h-4 w-4 shrink-0 text-muted-foreground" />
	{/if}

	<span class="truncate">{label}</span>

	<span class="ml-auto shrink-0 text-xs text-muted-foreground/70">{count}</span>
</button>
