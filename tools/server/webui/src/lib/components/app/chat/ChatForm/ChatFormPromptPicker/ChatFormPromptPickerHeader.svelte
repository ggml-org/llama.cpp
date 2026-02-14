<script lang="ts">
	import Badge from '$lib/components/ui/badge/badge.svelte';
	import type { MCPPromptInfo, MCPServerSettingsEntry } from '$lib/types';
	import { getFaviconUrl } from '$lib/utils';

	interface Props {
		prompt: MCPPromptInfo;
		server: MCPServerSettingsEntry | undefined;
		serverLabel: string;
	}

	let { prompt, server, serverLabel }: Props = $props();

	let faviconUrl = $derived(server ? getFaviconUrl(server.url) : null);
</script>

<div class="flex items-start gap-3">
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
				{prompt.title || prompt.name}
			</span>

			{#if prompt.arguments?.length}
				<Badge variant="secondary">
					{prompt.arguments.length} arg{prompt.arguments.length > 1 ? 's' : ''}
				</Badge>
			{/if}
		</div>

		{#if prompt.description}
			<p class="mt-1 text-sm text-muted-foreground">
				{prompt.description}
			</p>
		{/if}
	</div>
</div>
