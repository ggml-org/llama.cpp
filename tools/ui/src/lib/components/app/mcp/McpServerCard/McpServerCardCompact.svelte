<script lang="ts">
	import * as Card from '$lib/components/ui/card';
	import { Badge } from '$lib/components/ui/badge';
	import { Skeleton } from '$lib/components/ui/skeleton';
	import { Switch } from '$lib/components/ui/switch';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { McpServerIdentity } from '$lib/components/app/mcp';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { HealthCheckStatus } from '$lib/enums';
	import type { RecommendedMCPServer, HealthCheckState, MCPToolInfo } from '$lib/types';
	import { onMount } from 'svelte';

	interface Props {
		server: RecommendedMCPServer;
		enabled?: boolean;
		onToggle?: (enabled: boolean) => void;
	}

	let { server, enabled = false, onToggle }: Props = $props();

	onMount(() => {
		const state = mcpStore.getHealthCheckState(server.id);

		if (state.status === HealthCheckStatus.IDLE) {
			mcpStore.runHealthCheck(server).catch(() => {});
		}
	});

	let healthState = $derived<HealthCheckState>(mcpStore.getHealthCheckState(server.id));
	let displayName = $derived(mcpStore.getServerLabel(server));
	let faviconUrl = $derived(mcpStore.getServerFavicon(server.id));
	let isIdle = $derived(healthState.status === HealthCheckStatus.IDLE);
	let isHealthChecking = $derived(healthState.status === HealthCheckStatus.CONNECTING);
	let isConnected = $derived(healthState.status === HealthCheckStatus.SUCCESS);
	let isError = $derived(healthState.status === HealthCheckStatus.ERROR);
	let errorMessage = $derived(
		healthState.status === HealthCheckStatus.ERROR ? healthState.message : undefined
	);
	let serverInfo = $derived(
		healthState.status === HealthCheckStatus.SUCCESS ? healthState.serverInfo : undefined
	);
	let tools = $derived<MCPToolInfo[]>(
		healthState.status === HealthCheckStatus.SUCCESS ? healthState.tools : []
	);
	let showSkeleton = $derived(isIdle || isHealthChecking);
	let description = $derived(server.description);

	function handleToggle(checked: boolean) {
		onToggle?.(checked);
	}
</script>

<Card.Root class="!gap-3 bg-muted/30 p-4">
	<div class="flex items-start justify-between gap-3">
		<div class="min-w-0 flex-1">
			<McpServerIdentity
				{displayName}
				{faviconUrl}
				{serverInfo}
				iconClass="h-5 w-5"
				iconRounded="rounded"
				nameClass="font-medium"
			/>
		</div>

		<Switch checked={enabled} disabled={isError} onCheckedChange={handleToggle} />
	</div>

	{#if isError && errorMessage}
		<p class="text-xs text-destructive">{errorMessage}</p>
	{/if}

	{#if description && !showSkeleton}
		<p class="line-clamp-2 text-xs text-muted-foreground">
			{description}
		</p>
	{/if}

	<!-- {#if showSkeleton}
		<div class="flex flex-wrap gap-1.5">
			<Skeleton class="h-5 w-16 rounded-full" />
			<Skeleton class="h-5 w-20 rounded-full" />
			<Skeleton class="h-5 w-14 rounded-full" />
		</div>
	{:else if tools.length > 0}
		<div class="space-y-1.5">
			<p class="text-xs font-medium text-muted-foreground">Tools</p>

			<div class="flex flex-wrap gap-1.5">
				{#each tools as tool (tool.name)}
					<Tooltip.Root>
						<Tooltip.Trigger>
							<Badge variant="secondary" class="h-5 max-w-40 truncate px-2 text-[11px]">
								{tool.name}
							</Badge>
						</Tooltip.Trigger>

						<Tooltip.Content>
							<p class="max-w-xs text-xs">
								{tool.description ?? 'No description'}
							</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/each}
			</div>
		</div>
	{/if} -->
</Card.Root>
