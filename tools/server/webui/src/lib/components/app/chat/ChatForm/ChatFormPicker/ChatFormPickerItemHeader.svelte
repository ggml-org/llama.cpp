<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { MCPServerSettingsEntry } from '$lib/types';
	import { getFaviconUrl } from '$lib/utils';

	interface Props {
		server: MCPServerSettingsEntry | undefined;
		serverLabel: string;
		title: string;
		description?: string;
		titleExtra?: Snippet;
		subtitle?: Snippet;
	}

	let { server, serverLabel, title, description, titleExtra, subtitle }: Props = $props();

	let faviconUrl = $derived(server ? getFaviconUrl(server.url) : null);
</script>

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
			{title}
		</span>

		{#if titleExtra}
			{@render titleExtra()}
		{/if}
	</div>

	{#if description}
		<p class="mt-0.5 truncate text-sm text-muted-foreground">
			{description}
		</p>
	{/if}

	{#if subtitle}
		{@render subtitle()}
	{/if}
</div>
