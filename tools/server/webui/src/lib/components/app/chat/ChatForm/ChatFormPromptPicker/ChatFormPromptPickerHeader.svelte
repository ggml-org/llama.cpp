<script lang="ts">
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
	{#if faviconUrl}
		<img
			src={faviconUrl}
			alt=""
			class="mt-0.5 h-5 w-5 shrink-0 rounded"
			onerror={(e) => {
				(e.currentTarget as HTMLImageElement).style.display = 'none';
			}}
		/>
	{/if}

	<div class="min-w-0 flex-1">
		<div class="text-xs text-muted-foreground">
			{serverLabel}
		</div>

		<div class="flex items-center gap-2">
			<span class="font-medium">
				{prompt.title || prompt.name}
			</span>

			{#if prompt.arguments?.length}
				<span class="rounded-full bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
					{prompt.arguments.length} arg{prompt.arguments.length > 1 ? 's' : ''}
				</span>
			{/if}
		</div>
		{#if prompt.description}
			<p class="mt-1 text-sm text-muted-foreground">
				{prompt.description}
			</p>
		{/if}
	</div>
</div>
