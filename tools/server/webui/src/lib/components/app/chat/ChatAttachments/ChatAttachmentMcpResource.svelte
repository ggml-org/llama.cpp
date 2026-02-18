<script lang="ts">
	import { Loader2, AlertCircle } from '@lucide/svelte';
	import { cn } from '$lib/components/ui/utils';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import type { MCPResourceAttachment } from '$lib/types';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ActionIconRemove } from '$lib/components/app';
	import { getResourceIcon } from '$lib/utils';

	interface Props {
		attachment: MCPResourceAttachment;
		onRemove?: (attachmentId: string) => void;
		onClick?: () => void;
		class?: string;
	}

	let { attachment, onRemove, onClick, class: className }: Props = $props();

	function getStatusClass(attachment: MCPResourceAttachment): string {
		if (attachment.error) return 'border-red-500/50 bg-red-500/10';
		if (attachment.loading) return 'border-border/50 bg-muted/30';
		return 'border-border/50 bg-muted/30';
	}

	const ResourceIcon = $derived(
		getResourceIcon(attachment.resource.mimeType, attachment.resource.uri)
	);
	const serverName = $derived(mcpStore.getServerDisplayName(attachment.resource.serverName));
	const favicon = $derived(mcpStore.getServerFavicon(attachment.resource.serverName));
</script>

<Tooltip.Root>
	<Tooltip.Trigger>
		<button
			type="button"
			class={cn(
				'flex flex-shrink-0 items-center gap-1.5 rounded-md border px-2 py-0.75 text-sm transition-colors',
				getStatusClass(attachment),
				onClick && 'cursor-pointer hover:bg-muted/50',
				className
			)}
			onclick={onClick}
			disabled={!onClick}
		>
			{#if attachment.loading}
				<Loader2 class="h-3 w-3 animate-spin text-muted-foreground" />
			{:else if attachment.error}
				<AlertCircle class="h-3 w-3 text-red-500" />
			{:else}
				<ResourceIcon class="h-3 w-3 text-muted-foreground" />
			{/if}

			<span class="max-w-[150px] truncate text-xs">
				{attachment.resource.title || attachment.resource.name}
			</span>

			{#if onRemove}
				<ActionIconRemove class="-mr-1.5 bg-transparent -my-2" iconSize={2} id={attachment.id} {onRemove} />
			{/if}
		</button>
	</Tooltip.Trigger>

	<Tooltip.Content>
		<div class="flex items-center gap-1 text-xs">
			{#if favicon}
				<img
					src={favicon}
					alt=""
					class="h-3 w-3 shrink-0 rounded-sm"
					onerror={(e) => {
						(e.currentTarget as HTMLImageElement).style.display = 'none';
					}}
				/>
			{/if}

			<span class="truncate">
				{serverName}
			</span>
		</div>
	</Tooltip.Content>
</Tooltip.Root>
