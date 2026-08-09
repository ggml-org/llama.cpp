<script lang="ts">
	import { Folder, X } from '@lucide/svelte';
	import { ActionIcon } from '$lib/components/app/actions';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { SET_WORKING_DIRECTORY_LABEL } from '$lib/constants';
	import { abbreviateWorkingDir } from '$lib/utils';

	interface Props {
		directory?: string | null;
		homeBase?: string | null;
		disabled?: boolean;
		showTooltip?: boolean;
		onClear?: (event?: MouseEvent) => void;
	}

	let {
		directory = null,
		disabled = false,
		homeBase = null,
		onClear,
		showTooltip = false
	}: Props = $props();

	const displayLabel = $derived(
		directory ? abbreviateWorkingDir(directory, homeBase) : SET_WORKING_DIRECTORY_LABEL
	);
	// Full path surface: hover the abbreviated label to recall the exact directory.
	const displayLabelTitle = $derived(directory ?? '');
</script>

<span
	class="text-muted-foreground inline-flex items-center gap-1 text-xs group"
	class:text-foreground={directory}
>
	<div class="flex min-w-0 items-center gap-1 cursor-pointer">
		<Folder class="w-3.5 h-3.5" />

		{#if showTooltip && displayLabelTitle}
			<Tooltip.Root>
				<Tooltip.Trigger>
					{#snippet child({ props })}
						<span {...props} class="max-w-64 truncate">{displayLabel}</span>
					{/snippet}
				</Tooltip.Trigger>
				<Tooltip.Content>
					<p>{displayLabelTitle}</p>
				</Tooltip.Content>
			</Tooltip.Root>
		{:else}
			<span class="max-w-64 truncate">{displayLabel}</span>
		{/if}
	</div>

	{#if directory}
		<div
			class="w-0 overflow-hidden opacity-0 transition-[width,opacity] duration-200 ease-out group-hover:w-auto group-hover:opacity-100"
		>
			<ActionIcon
				icon={X}
				tooltip="Reset working directory"
				ariaLabel="Reset working directory"
				{disabled}
				onclick={onClear}
				iconSize="h-3 w-3"
				stopPropagationOnClick
				class="!h-4 !w-4 shrink-0 text-muted-foreground hover:text-foreground"
			/>
		</div>
	{/if}
</span>
