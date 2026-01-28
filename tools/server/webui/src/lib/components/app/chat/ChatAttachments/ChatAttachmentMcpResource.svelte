<script lang="ts">
	import { FileText, X, Loader2, AlertCircle, Database, Image, Code, File } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { cn } from '$lib/components/ui/utils';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { getFaviconUrl } from '$lib/utils';
	import type { MCPResourceAttachment, MCPResourceInfo } from '$lib/types';
	import * as Tooltip from '$lib/components/ui/tooltip';

	interface Props {
		attachment: MCPResourceAttachment;
		onRemove?: (attachmentId: string) => void;
		onClick?: () => void;
		class?: string;
	}

	let { attachment, onRemove, onClick, class: className }: Props = $props();

	function getResourceIcon(resource: MCPResourceInfo) {
		const mimeType = resource.mimeType?.toLowerCase() || '';
		const uri = resource.uri.toLowerCase();

		if (mimeType.startsWith('image/') || /\.(png|jpg|jpeg|gif|svg|webp)$/.test(uri)) {
			return Image;
		}
		if (
			mimeType.includes('json') ||
			mimeType.includes('javascript') ||
			mimeType.includes('typescript') ||
			/\.(js|ts|json|yaml|yml|xml|html|css)$/.test(uri)
		) {
			return Code;
		}
		if (mimeType.includes('text') || /\.(txt|md|log)$/.test(uri)) {
			return FileText;
		}
		if (uri.includes('database') || uri.includes('db://')) {
			return Database;
		}
		return File;
	}

	function getStatusClass(attachment: MCPResourceAttachment): string {
		if (attachment.error) return 'border-red-500/50 bg-red-500/10';
		if (attachment.loading) return 'border-border/50 bg-muted/30';
		return 'border-border/50 bg-muted/30';
	}

	function getServerDisplayName(serverId: string): string {
		const server = mcpStore.getServerById(serverId);
		return server ? mcpStore.getServerLabel(server) : serverId;
	}

	function getServerFavicon(serverId: string): string | null {
		const server = mcpStore.getServerById(serverId);
		return server ? getFaviconUrl(server.url) : null;
	}

	const ResourceIcon = $derived(getResourceIcon(attachment.resource));
	const serverName = $derived(getServerDisplayName(attachment.resource.serverName));
	const favicon = $derived(getServerFavicon(attachment.resource.serverName));
</script>

<Tooltip.Root>
	<Tooltip.Trigger>
		<button
			type="button"
			class={cn(
				'flex flex-shrink-0 items-center gap-2 rounded-md border px-2 py-1 text-sm transition-colors',
				getStatusClass(attachment),
				onClick && 'cursor-pointer hover:bg-muted/50',
				className
			)}
			onclick={onClick}
			disabled={!onClick}
		>
			{#if attachment.loading}
				<Loader2 class="h-3.5 w-3.5 animate-spin text-muted-foreground" />
			{:else if attachment.error}
				<AlertCircle class="h-3.5 w-3.5 text-red-500" />
			{:else}
				<ResourceIcon class="h-3.5 w-3.5 text-muted-foreground" />
			{/if}

			<span class="max-w-[150px] truncate">
				{attachment.resource.title || attachment.resource.name}
			</span>

			{#if onRemove}
				<Button
					variant="ghost"
					size="sm"
					class="h-5 w-5 p-0 hover:bg-destructive/20"
					onclick={(e) => {
						e.stopPropagation();
						onRemove(attachment.id);
					}}
					title="Remove attachment"
				>
					<X class="h-3 w-3" />
				</Button>
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
