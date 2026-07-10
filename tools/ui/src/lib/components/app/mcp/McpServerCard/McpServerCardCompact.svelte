<script lang="ts">
	import { X } from '@lucide/svelte';
	import * as Card from '$lib/components/ui/card';
	import { Badge } from '$lib/components/ui/badge';
	import { Skeleton } from '$lib/components/ui/skeleton';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { McpServerIdentity } from '$lib/components/app/mcp';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { HealthCheckStatus } from '$lib/enums';
	import type { HealthCheckState, MCPServerSettingsEntry } from '$lib/types';
	import { onMount } from 'svelte';
	import { MCP_CARD_VISIBLE_TOOL_LIMIT, NEWLINE } from '$lib/constants';

	interface Props {
		server: MCPServerSettingsEntry & { description?: string };
		onClick?: () => void;
		onDismiss?: () => void;
		selected?: boolean;
		dimmed?: boolean;
	}

	let { server, onClick, onDismiss, selected = false, dimmed = false }: Props = $props();

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
	let isError = $derived(healthState.status === HealthCheckStatus.ERROR);
	let errorMessage = $derived(
		healthState.status === HealthCheckStatus.ERROR ? healthState.message : undefined
	);
	let serverInfo = $derived(
		healthState.status === HealthCheckStatus.SUCCESS ? healthState.serverInfo : undefined
	);
	let tools = $derived(healthState.status === HealthCheckStatus.SUCCESS ? healthState.tools : []);
	let instructions = $derived(
		healthState.status === HealthCheckStatus.SUCCESS ? healthState.instructions : undefined
	);
	let showSkeleton = $derived(isIdle || isHealthChecking);

	// Curated descriptions get two lines; instructions fallback is one line so the
	// compact card stays scannable.
	let description = $derived.by(() => {
		if (server.description) {
			return { text: server.description, lines: 2 };
		}
		if (!instructions) return null;
		const firstLine = instructions.split(NEWLINE).find((line: string) => line.trim().length > 0);
		const trimmed = firstLine?.trim();
		return trimmed ? { text: trimmed, lines: 1 } : null;
	});

	let visibleTools = $derived(tools.slice(0, MCP_CARD_VISIBLE_TOOL_LIMIT));
	let hiddenTools = $derived(tools.slice(MCP_CARD_VISIBLE_TOOL_LIMIT));
	let hiddenToolCount = $derived(hiddenTools.length);

	function handleDismissClick(event: MouseEvent) {
		event.stopPropagation();
		onDismiss?.();
	}
</script>

<Card.Root
	class={`relative !gap-3 bg-muted/30 p-4 transition-colors ${onClick ? 'cursor-pointer hover:bg-muted/50' : ''} ${selected ? 'bg-background ring-2 ring-primary/40' : ''} ${dimmed ? 'opacity-50' : ''}`}
	onclick={onClick}
>
	{#if onDismiss}
		<button
			type="button"
			onclick={handleDismissClick}
			aria-label={`Dismiss ${displayName}`}
			class="absolute top-4 right-4 rounded-xs opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:outline-hidden disabled:pointer-events-none [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4"
		>
			<X />

			<span class="sr-only">Dismiss</span>
		</button>
	{/if}

	<div class={['min-w-0', onDismiss ? 'pr-7' : '']}>
		{#if showSkeleton}
			<span class="flex min-w-0 items-center gap-1.5">
				<Skeleton class="h-5 w-5 rounded" />
				<Skeleton class="h-4 w-32" />
			</span>
		{:else}
			<McpServerIdentity
				{displayName}
				{faviconUrl}
				{serverInfo}
				iconClass="h-5 w-5"
				iconRounded="rounded"
				nameClass="font-medium"
			/>
		{/if}
	</div>

	{#if isError && errorMessage}
		<p class="text-xs text-destructive">{errorMessage}</p>
	{/if}

	{#if showSkeleton}
		<div class="space-y-1.5">
			<Skeleton class="h-3 w-full max-w-md" />
		</div>

		<div class="flex flex-wrap items-center gap-1.5">
			<Skeleton class="h-5 w-16 rounded-full" />
			<Skeleton class="h-5 w-20 rounded-full" />
			<Skeleton class="h-5 w-24 rounded-full" />
			<Skeleton class="h-5 w-14 rounded-full" />
		</div>
	{:else}
		{#if description}
			{#if description.lines === 2}
				<p class="line-clamp-2 text-xs text-muted-foreground" title={description.text}>
					{description.text}
				</p>
			{:else}
				<p class="line-clamp-1 truncate text-xs text-muted-foreground" title={description.text}>
					{description.text}
				</p>
			{/if}
		{/if}

		{#if tools.length > 0}
			<div class="flex flex-wrap items-center gap-1.5">
				{#each visibleTools as tool (tool.name)}
					<Tooltip.Root>
						<Tooltip.Trigger>
							<Badge variant="secondary" class="h-5 max-w-40 px-2 text-[11px]">
								<span class="block min-w-0 flex-1 truncate">{tool.name}</span>
							</Badge>
						</Tooltip.Trigger>

						<Tooltip.Content>
							<p class="max-w-xs text-xs">
								{tool.description ?? 'No description'}
							</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/each}

				{#if hiddenToolCount > 0}
					<Tooltip.Root>
						<Tooltip.Trigger>
							<Badge variant="secondary" class="h-5 px-2 text-[11px] text-muted-foreground">
								+ {hiddenToolCount} more tools
							</Badge>
						</Tooltip.Trigger>

						<Tooltip.Content class="max-w-md">
							<p class="text-xs">
								{hiddenTools.map((tool) => tool.name).join(', ')}
							</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}
			</div>
		{/if}
	{/if}
</Card.Root>
