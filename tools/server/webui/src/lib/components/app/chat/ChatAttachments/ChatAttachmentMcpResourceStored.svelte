<script lang="ts">
	import { FileText, Database, Image, Code, File } from '@lucide/svelte';
	import { cn } from '$lib/components/ui/utils';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import type { DatabaseMessageExtraMcpResource } from '$lib/types';
	import {
		IMAGE_FILE_EXTENSION_REGEX,
		CODE_FILE_EXTENSION_REGEX,
		TEXT_FILE_EXTENSION_REGEX
	} from '$lib/constants/mcp-resource';
	import { MimeTypePrefix, MimeTypeIncludes, UriPattern } from '$lib/enums';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ActionIconRemove } from '$lib/components/app';

	interface Props {
		extra: DatabaseMessageExtraMcpResource;
		readonly?: boolean;
		onRemove?: () => void;
		onClick?: (event?: MouseEvent) => void;
		class?: string;
	}

	let { extra, readonly = true, onRemove, onClick, class: className }: Props = $props();

	function getResourceIcon(mimeType?: string, uri?: string) {
		const mime = mimeType?.toLowerCase() || '';
		const u = uri?.toLowerCase() || '';

		if (mime.startsWith(MimeTypePrefix.IMAGE) || IMAGE_FILE_EXTENSION_REGEX.test(u)) {
			return Image;
		}

		if (
			mime.includes(MimeTypeIncludes.JSON) ||
			mime.includes(MimeTypeIncludes.JAVASCRIPT) ||
			mime.includes(MimeTypeIncludes.TYPESCRIPT) ||
			CODE_FILE_EXTENSION_REGEX.test(u)
		) {
			return Code;
		}

		if (mime.includes(MimeTypePrefix.TEXT) || TEXT_FILE_EXTENSION_REGEX.test(u)) {
			return FileText;
		}

		if (u.includes(UriPattern.DATABASE_KEYWORD) || u.includes(UriPattern.DATABASE_SCHEME)) {
			return Database;
		}

		return File;
	}

	const ResourceIcon = $derived(getResourceIcon(extra.mimeType, extra.uri));
	const serverName = $derived(mcpStore.getServerDisplayName(extra.serverName));
	const favicon = $derived(mcpStore.getServerFavicon(extra.serverName));
</script>

<Tooltip.Root>
	<Tooltip.Trigger>
		<button
			type="button"
			class={cn(
				'flex flex-shrink-0 items-center gap-2 rounded-md border border-border/50 bg-muted/30 p-0.5 px-2 text-sm',
				onClick && 'cursor-pointer hover:bg-muted/50',
				className
			)}
			onclick={(e) => {
				e.stopPropagation();
				onClick?.(e);
			}}
			disabled={!onClick}
		>
			<ResourceIcon class="h-3.5 w-3.5 text-muted-foreground" />

			<span class="max-w-[150px] truncate">
				{extra.name}
			</span>

			{#if !readonly && onRemove}
				<ActionIconRemove class="bg-transparent" id={extra.uri} onRemove={() => onRemove?.()} />
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
