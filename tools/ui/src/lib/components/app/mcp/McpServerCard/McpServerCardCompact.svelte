<script lang="ts">
	import * as Card from '$lib/components/ui/card';
	import { mode } from 'mode-watcher';
	import type { RecommendedMCPServer } from '$lib/types';

	interface Props {
		server: RecommendedMCPServer;
		onClick?: () => void;
		selected?: boolean;
		dimmed?: boolean;
	}

	let { server, onClick, selected = false, dimmed = false }: Props = $props();

	let activeIconUrl = $derived.by(() => {
		const isDark = mode.current === 'dark';

		if (isDark && server.iconUrlDark) return server.iconUrlDark;
		if (!isDark && server.iconUrlLight) return server.iconUrlLight;

		return server.iconUrl;
	});
</script>

<Card.Root
	class={`relative gap-3! select-none bg-muted/30 p-4 transition-all ${onClick ? 'cursor-pointer hover:bg-muted/50 hover:opacity-100' : ''} ${selected ? 'bg-muted/30 ring-1 ring-primary/40' : ''} ${dimmed ? 'opacity-50' : ''}`}
	onclick={onClick}
>
	<div class="flex min-w-0 items-center gap-2">
		{#if activeIconUrl}
			<img
				src={activeIconUrl}
				alt=""
				class="h-5 w-5 shrink-0 rounded"
				loading="lazy"
				decoding="async"
			/>
		{/if}

		<h4 class="min-w-0 flex-1 truncate font-medium">{server.name}</h4>
	</div>

	<p class="text-xs text-muted-foreground">{server.description}</p>
</Card.Root>
