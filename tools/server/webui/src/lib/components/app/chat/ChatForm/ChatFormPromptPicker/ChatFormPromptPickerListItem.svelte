<script lang="ts">
	import type { MCPPromptInfo, MCPServerSettingsEntry } from '$lib/types';
	import { getFaviconUrl } from '$lib/utils';

	interface Props {
		prompt: MCPPromptInfo;
		server: MCPServerSettingsEntry | undefined;
		serverLabel: string;
		isSelected?: boolean;
		onClick: () => void;
	}

	let { prompt, server, serverLabel, isSelected = false, onClick }: Props = $props();

	let faviconUrl = $derived(server ? getFaviconUrl(server.url) : null);
</script>

<button
	type="button"
	onclick={onClick}
	class="flex w-full cursor-pointer items-start gap-3 rounded-lg px-3 py-2 text-left hover:bg-accent {isSelected
		? 'bg-accent'
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
			<span class="font-medium">{prompt.title || prompt.name}</span>

			{#if prompt.arguments && prompt.arguments.length > 0}
				<span class="text-xs text-muted-foreground">
					({prompt.arguments.length} arg{prompt.arguments.length > 1 ? 's' : ''})
				</span>
			{/if}
		</div>

		{#if prompt.description}
			<p class="truncate text-sm text-muted-foreground">
				{prompt.description}
			</p>
		{/if}
	</div>
</button>
