<script lang="ts">
	import { Cable, ExternalLink } from '@lucide/svelte';
	import { Switch } from '$lib/components/ui/switch';
	import type { MCPServerInfo, MCPCapabilitiesInfo } from '$lib/types';
	import { Badge } from '$lib/components/ui/badge';
	import McpCapabilitiesBadges from './McpCapabilitiesBadges.svelte';

	interface Props {
		displayName: string;
		faviconUrl: string | null;
		enabled: boolean;
		onToggle: (enabled: boolean) => void;
		serverInfo?: MCPServerInfo;
		capabilities?: MCPCapabilitiesInfo;
	}

	let { displayName, faviconUrl, enabled, onToggle, serverInfo, capabilities }: Props = $props();
</script>

<div class="space-y-3">
	<div class="flex items-start justify-between gap-3">
		<div class="grid min-w-0 gap-3">
			<div class="flex items-center gap-2">
				{#if faviconUrl}
					<img
						src={faviconUrl}
						alt=""
						class="h-5 w-5 shrink-0 rounded"
						onerror={(e) => {
							(e.currentTarget as HTMLImageElement).style.display = 'none';
						}}
					/>
				{:else}
					<div class="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-muted">
						<Cable class="h-3 w-3 text-muted-foreground" />
					</div>
				{/if}

				<p class="truncate leading-none font-medium">
					{serverInfo?.title || serverInfo?.name || displayName}
				</p>

				{#if serverInfo?.version}
					<Badge variant="secondary" class="h-4 shrink-0 px-1 text-[10px]">
						v{serverInfo.version}
					</Badge>
				{/if}

				{#if serverInfo?.websiteUrl}
					<a
						href={serverInfo.websiteUrl}
						target="_blank"
						rel="noopener noreferrer"
						class="shrink-0 text-muted-foreground hover:text-foreground"
						aria-label="Open website"
					>
						<ExternalLink class="h-3 w-3" />
					</a>
				{/if}
			</div>

			{#if capabilities}
				<McpCapabilitiesBadges {capabilities} />
			{/if}
		</div>

		<div class="flex shrink-0 items-center pl-2">
			<Switch checked={enabled} onCheckedChange={onToggle} />
		</div>
	</div>
</div>
