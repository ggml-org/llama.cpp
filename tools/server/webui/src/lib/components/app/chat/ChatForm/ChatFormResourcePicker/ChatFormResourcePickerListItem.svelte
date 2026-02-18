<script lang="ts">
	import type { MCPResourceInfo, MCPServerSettingsEntry } from '$lib/types';
	import { getFaviconUrl } from '$lib/utils';

	interface Props {
		resource: MCPResourceInfo;
		server: MCPServerSettingsEntry | undefined;
		serverLabel: string;
		isSelected?: boolean;
		isAttached?: boolean;
		onClick: () => void;
		'data-resource-index'?: number;
	}

	let {
		resource,
		server,
		serverLabel,
		isSelected = false,
		isAttached = false,
		onClick,
		'data-resource-index': dataResourceIndex
	}: Props = $props();

	let faviconUrl = $derived(server ? getFaviconUrl(server.url) : null);
</script>

<button
	type="button"
	data-resource-index={dataResourceIndex}
	onclick={onClick}
	class="flex w-full cursor-pointer items-start gap-3 rounded-lg px-3 py-2 text-left hover:bg-accent/50 {isSelected
		? 'bg-accent/50'
		: ''}"
>
	<div class="min-w-0 flex-1">
		<div class="mb-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
			{#if faviconUrl}
				<img
					src={faviconUrl}
					alt=""
					class="h-3 w-3 shrink-0 rounded-sm"
					onerror={(e) => {
						(e.currentTarget as HTMLImageElement).style.display = 'none';
					}}
				/>
			{/if}

			<span>{serverLabel}</span>
		</div>

		<div class="flex items-center gap-2">
			<span class="font-medium">
				{resource.title || resource.name}
			</span>

			{#if isAttached}
				<span
					class="inline-flex items-center rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary"
				>
					attached
				</span>
			{/if}
		</div>

		{#if resource.description}
			<p class="mt-0.5 truncate text-sm text-muted-foreground">
				{resource.description}
			</p>
		{/if}

		<p class="mt-0.5 truncate text-xs text-muted-foreground/60">
			{resource.uri}
		</p>
	</div>
</button>
