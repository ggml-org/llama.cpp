<script lang="ts">
	import { onMount } from 'svelte';
	import { ChevronDown, Settings } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { Switch } from '$lib/components/ui/switch';
	import { cn } from '$lib/components/ui/utils';
	import { SearchableDropdownMenu } from '$lib/components/app';
	import McpLogo from '$lib/components/app/misc/McpLogo.svelte';
	import { settingsStore } from '$lib/stores/settings.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { parseMcpServerSettings, parseMcpServerUsageStats } from '$lib/config/mcp';
	import type { MCPServerSettingsEntry } from '$lib/types/mcp';
	import {
		mcpGetHealthCheckState,
		mcpHasHealthCheck,
		mcpRunHealthCheck
	} from '$lib/stores/mcp.svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		onSettingsClick?: () => void;
	}

	let { class: className = '', disabled = false, onSettingsClick }: Props = $props();

	let searchQuery = $state('');

	let mcpServers = $derived.by(() => {
		return parseMcpServerSettings(settingsStore.config.mcpServers);
	});

	let hasMcpServers = $derived(mcpServers.length > 0);

	let mcpUsageStats = $derived(parseMcpServerUsageStats(settingsStore.config.mcpServerUsageStats));

	function getServerUsageCount(serverId: string): number {
		return mcpUsageStats[serverId] || 0;
	}

	function isServerEnabledForChat(server: MCPServerSettingsEntry): boolean {
		return conversationsStore.isMcpServerEnabledForChat(server.id, server.enabled);
	}

	function hasPerChatOverride(serverId: string): boolean {
		return conversationsStore.getMcpServerOverride(serverId) !== undefined;
	}

	let enabledMcpServersForChat = $derived(
		mcpServers.filter((s) => isServerEnabledForChat(s) && s.url.trim())
	);

	let healthyEnabledMcpServers = $derived(
		enabledMcpServersForChat.filter((s) => {
			const healthState = mcpGetHealthCheckState(s.id);
			return healthState.status !== 'error';
		})
	);

	let hasEnabledMcpServers = $derived(enabledMcpServersForChat.length > 0);

	let sortedMcpServers = $derived(
		[...mcpServers].sort((a, b) => {
			// First: globally enabled servers come first
			if (a.enabled !== b.enabled) return a.enabled ? -1 : 1;

			// Then sort by usage count (descending)
			const usageA = getServerUsageCount(a.id);
			const usageB = getServerUsageCount(b.id);
			if (usageB !== usageA) return usageB - usageA;

			// Then alphabetically by name
			return getServerDisplayName(a).localeCompare(getServerDisplayName(b));
		})
	);

	let filteredMcpServers = $derived(() => {
		const query = searchQuery.toLowerCase().trim();
		if (query) {
			return sortedMcpServers.filter((s) => {
				const name = getServerDisplayName(s).toLowerCase();
				const url = s.url.toLowerCase();
				return name.includes(query) || url.includes(query);
			});
		}

		return sortedMcpServers.slice(0, 4);
	});

	let extraServersCount = $derived(Math.max(0, healthyEnabledMcpServers.length - 3));

	async function toggleServerForChat(serverId: string, globalEnabled: boolean) {
		await conversationsStore.toggleMcpServerForChat(serverId, globalEnabled);
	}

	function getServerDisplayName(server: MCPServerSettingsEntry): string {
		if (server.name) return server.name;
		try {
			const url = new URL(server.url);
			const host = url.hostname.replace(/^(www\.|mcp\.)/, '');
			const name = host.split('.')[0] || 'Unknown';
			return name.charAt(0).toUpperCase() + name.slice(1);
		} catch {
			return 'New Server';
		}
	}

	function getFaviconUrl(server: MCPServerSettingsEntry): string | null {
		try {
			const url = new URL(server.url);
			const hostnameParts = url.hostname.split('.');
			const rootDomain =
				hostnameParts.length >= 2 ? hostnameParts.slice(-2).join('.') : url.hostname;
			return `https://www.google.com/s2/favicons?domain=${rootDomain}&sz=32`;
		} catch {
			return null;
		}
	}

	let mcpFavicons = $derived(
		healthyEnabledMcpServers
			.slice(0, 3)
			.map((s) => ({ id: s.id, url: getFaviconUrl(s) }))
			.filter((f) => f.url !== null)
	);

	let serversWithUrls = $derived(mcpServers.filter((s) => s.url.trim()));

	onMount(() => {
		for (const server of serversWithUrls) {
			if (!mcpHasHealthCheck(server.id)) {
				mcpRunHealthCheck(server);
			}
		}
	});
</script>

{#if hasMcpServers}
	<SearchableDropdownMenu
		bind:searchValue={searchQuery}
		placeholder="Search servers..."
		emptyMessage="No servers found"
		isEmpty={filteredMcpServers().length === 0}
		{disabled}
	>
		{#snippet trigger()}
			<button
				type="button"
				class={cn(
					'inline-flex cursor-pointer items-center rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60',
					hasEnabledMcpServers ? 'text-foreground' : 'text-muted-foreground',
					className
				)}
				{disabled}
				aria-label="MCP Servers"
			>
				<McpLogo style="width: 0.875rem; height: 0.875rem;" />

				<span class="mx-1.5 font-medium">MCP</span>

				{#if hasEnabledMcpServers && mcpFavicons.length > 0}
					<div class="flex -space-x-1">
						{#each mcpFavicons as favicon (favicon.id)}
							<img
								src={favicon.url}
								alt=""
								class="h-3.5 w-3.5 rounded-sm"
								onerror={(e) => {
									(e.currentTarget as HTMLImageElement).style.display = 'none';
								}}
							/>
						{/each}
					</div>

					{#if extraServersCount > 0}
						<span class="ml-1 text-muted-foreground">+{extraServersCount}</span>
					{/if}
				{/if}

				<ChevronDown class="h-3 w-3.5" />
			</button>
		{/snippet}

		{#each filteredMcpServers() as server (server.id)}
			{@const healthState = mcpGetHealthCheckState(server.id)}
			{@const hasError = healthState.status === 'error'}
			{@const isEnabledForChat = isServerEnabledForChat(server)}
			{@const hasOverride = hasPerChatOverride(server.id)}

			<div class="flex items-center justify-between gap-2 px-2 py-2">
				<div class="flex min-w-0 flex-1 items-center gap-2">
					{#if getFaviconUrl(server)}
						<img
							src={getFaviconUrl(server)}
							alt=""
							class="h-4 w-4 shrink-0 rounded-sm"
							onerror={(e) => {
								(e.currentTarget as HTMLImageElement).style.display = 'none';
							}}
						/>
					{/if}
					<span class="truncate text-sm">{getServerDisplayName(server)}</span>
					{#if hasError}
						<span class="shrink-0 rounded bg-destructive/15 px-1.5 py-0.5 text-xs text-destructive"
							>Error</span
						>
					{:else if server.enabled}
						<span class="shrink-0 rounded bg-primary/15 px-1.5 py-0.5 text-xs text-primary"
							>Global</span
						>
					{/if}
				</div>
				<Switch
					checked={isEnabledForChat}
					onCheckedChange={() => toggleServerForChat(server.id, server.enabled)}
					disabled={hasError}
					class={hasOverride ? 'ring-2 ring-primary/50 ring-offset-1' : ''}
				/>
			</div>
		{/each}

		{#snippet footer()}
			<DropdownMenu.Item class="flex cursor-pointer items-center gap-2" onclick={onSettingsClick}>
				<Settings class="h-4 w-4" />
				<span>Manage MCP Servers</span>
			</DropdownMenu.Item>
		{/snippet}
	</SearchableDropdownMenu>
{:else}
	<button
		type="button"
		class={cn(
			'inline-flex cursor-pointer items-center rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60',
			'text-muted-foreground',
			className
		)}
		{disabled}
		aria-label="MCP Servers"
		onclick={onSettingsClick}
	>
		<McpLogo style="width: 0.875rem; height: 0.875rem;" />
		<span class="mx-1.5 font-medium">MCP</span>
	</button>
{/if}
