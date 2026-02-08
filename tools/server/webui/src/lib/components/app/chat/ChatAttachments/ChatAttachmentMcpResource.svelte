<script lang="ts">
	import { FileText, Loader2, AlertCircle, Database, Image, Code, File } from '@lucide/svelte';
	import { cn } from '$lib/components/ui/utils';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import type { MCPResourceAttachment, MCPResourceInfo } from '$lib/types';
	import {
		IMAGE_FILE_EXTENSION_REGEX,
		CODE_FILE_EXTENSION_REGEX,
		TEXT_FILE_EXTENSION_REGEX
	} from '$lib/constants/mcp-resource';
	import { MimeTypePrefix, MimeTypeIncludes, UriPattern } from '$lib/enums';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ActionIconRemove } from '$lib/components/app';

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

		if (mimeType.startsWith(MimeTypePrefix.IMAGE) || IMAGE_FILE_EXTENSION_REGEX.test(uri)) {
			return Image;
		}
		if (
			mimeType.includes(MimeTypeIncludes.JSON) ||
			mimeType.includes(MimeTypeIncludes.JAVASCRIPT) ||
			mimeType.includes(MimeTypeIncludes.TYPESCRIPT) ||
			CODE_FILE_EXTENSION_REGEX.test(uri)
		) {
			return Code;
		}
		if (mimeType.includes(MimeTypePrefix.TEXT) || TEXT_FILE_EXTENSION_REGEX.test(uri)) {
			return FileText;
		}
		if (uri.includes(UriPattern.DATABASE_KEYWORD) || uri.includes(UriPattern.DATABASE_SCHEME)) {
			return Database;
		}
		return File;
	}

	function getStatusClass(attachment: MCPResourceAttachment): string {
		if (attachment.error) return 'border-red-500/50 bg-red-500/10';
		if (attachment.loading) return 'border-border/50 bg-muted/30';
		return 'border-border/50 bg-muted/30';
	}

	const ResourceIcon = $derived(getResourceIcon(attachment.resource));
	const serverName = $derived(mcpStore.getServerDisplayName(attachment.resource.serverName));
	const favicon = $derived(mcpStore.getServerFavicon(attachment.resource.serverName));
</script>

<Tooltip.Root>
	<Tooltip.Trigger>
		<button
			type="button"
			class={cn(
				'flex flex-shrink-0 items-center gap-2 rounded-md border p-0.5 pl-2 text-sm transition-colors',
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
				<ActionIconRemove class="bg-transparent " id={attachment.id} {onRemove} />
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
