<script lang="ts">
	import { Loader2, AlertCircle } from '@lucide/svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import type { MCPResourceAttachment } from '$lib/types';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ActionIcon } from '$lib/components/app';
	import { X } from '@lucide/svelte';
	import { getResourceIcon, getResourceDisplayName } from '$lib/utils';

	interface Props {
		attachment: MCPResourceAttachment;
		class?: string;
		onclick?: () => void;
		onRemove?: (attachmentId: string) => void;
	}

	let { attachment, class: className, onclick, onRemove }: Props = $props();

	const ResourceIcon = $derived(
		getResourceIcon(attachment.resource.mimeType, attachment.resource.uri)
	);
	const serverName = $derived(mcpStore.getServerDisplayName(attachment.resource.serverName));
	const favicon = $derived(mcpStore.getServerFavicon(attachment.resource.serverName));

	function getStatusClass(attachment: MCPResourceAttachment): string {
		if (attachment.error) return 'border-red-500/50 bg-red-500/10';
		if (attachment.loading) return 'border-border/50 bg-muted/30';

		return 'border-border/50 bg-muted/30';
	}
</script>

<Tooltip.Root>
	<Tooltip.Trigger>
		<button
			class={[
				'flex flex-shrink-0 items-center gap-1.5 rounded-md border px-2 py-0.75 text-sm transition-colors',
				getStatusClass(attachment),
				onclick && 'cursor-pointer hover:bg-muted/50',
				className
			]}
			disabled={!onclick}
			{onclick}
			type="button"
		>
			{#if attachment.loading}
				<Loader2 class="h-3 w-3 animate-spin text-muted-foreground" />
			{:else if attachment.error}
				<AlertCircle class="h-3 w-3 text-red-500" />
			{:else}
				<ResourceIcon class="h-3 w-3 text-muted-foreground" />
			{/if}

			<span class="max-w-[150px] truncate text-xs">
				{getResourceDisplayName(attachment.resource)}
			</span>

			{#if onRemove}
				<ActionIcon
					class="-my-2 -mr-1.5 bg-transparent"
					icon={X}
					iconSize="h-2 w-2"
					onclick={() => onRemove?.(attachment.id)}
					stopPropagationOnClick
					tooltip="Remove"
				/>
			{/if}
		</button>
	</Tooltip.Trigger>

	<Tooltip.Content>
		<div class="flex items-center gap-1 text-xs">
			{#if favicon}
				<img
					alt={attachment.resource.serverName}
					class="h-3 w-3 shrink-0 rounded-sm"
					onerror={(e) => {
						(e.currentTarget as HTMLImageElement).style.display = 'none';
					}}
					src={favicon}
				/>
			{/if}

			<span class="truncate">
				{serverName}
			</span>
		</div>
	</Tooltip.Content>
</Tooltip.Root>
