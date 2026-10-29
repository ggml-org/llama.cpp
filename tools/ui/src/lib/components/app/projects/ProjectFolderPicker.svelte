<script lang="ts">
	import { FolderPlus, X } from '@lucide/svelte';
	import {
		ActionIcon,
		ChatFormCurrentWorkingDirectory,
		ChatMessageCwdChangeRow
	} from '$lib/components/app';
	import { toolsStore } from '$lib/stores';
	import { abbreviateWorkingDir } from '$lib/utils';

	interface Props {
		cwd: string | null;
		onChange: (cwd: string | null) => void;
	}

	let { cwd, onChange }: Props = $props();

	// The row is the trigger, the picker's own chip stays hidden and its
	// popover anchors here instead.
	let anchor = $state<HTMLDivElement | null>(null);
	let isOpen = $state(false);
	let query = $state('');

	let homeBase = $derived(toolsStore.serverHome);
</script>

<div class="flex items-center gap-1">
	<div bind:this={anchor} class="min-w-0">
		{#if cwd}
			<button
				class="min-w-0 rounded-md px-1 text-left hover:bg-muted/60"
				onclick={() => (isOpen = true)}
				type="button"
			>
				<ChatMessageCwdChangeRow
					class="py-1"
					display={abbreviateWorkingDir(cwd, homeBase)}
					path={cwd}
				/>
			</button>
		{:else}
			<button
				class="flex items-center gap-1.5 rounded-md px-1 py-1.5 text-sm text-muted-foreground hover:bg-muted/60"
				onclick={() => (isOpen = true)}
				type="button"
			>
				<FolderPlus class="h-3.5 w-3.5 shrink-0" />
				Set working directory
			</button>
		{/if}
	</div>

	{#if cwd}
		<ActionIcon
			ariaLabel="Clear folder"
			class="h-5 w-5 shrink-0 text-muted-foreground"
			icon={X}
			iconSize="h-3.5 w-3.5"
			onclick={() => onChange(null)}
			tooltip="Clear folder"
			tooltipAsTitle
		/>
	{/if}
</div>

<ChatFormCurrentWorkingDirectory
	bind:query
	customAnchor={anchor}
	directory={cwd}
	{isOpen}
	onChange={(directory) => onChange(directory)}
	onClose={() => (isOpen = false)}
	showChip={false}
/>
