<script lang="ts">
	import { Settings, Plus } from '@lucide/svelte';
	import { Switch } from '$lib/components/ui/switch';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { McpLogo, DropdownMenuSearchable, McpServerIdentity } from '$lib/components/app';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { HealthCheckStatus } from '$lib/enums';
	import type { MCPServerSettingsEntry } from '$lib/types';
	import { goto } from '$app/navigation';

	interface Props {
		onMcpSettingsClick?: () => void;
	}

	let { onMcpSettingsClick }: Props = $props();

	let mcpSearchQuery = $state('');
	let allMcpServers = $derived(mcpStore.getServersSorted());
	let mcpServers = $derived(allMcpServers.filter((s) => s.enabled));
	let hasMcpServers = $derived(mcpServers.length > 0);
	// let hasAnyMcpServers = $derived(allMcpServers.length > 0);
	let filteredMcpServers = $derived.by(() => {
		const query = mcpSearchQuery.toLowerCase().trim();
		if (!query) return mcpServers;
		return mcpServers.filter((s) => {
			const name = getServerLabel(s).toLowerCase();
			const url = s.url.toLowerCase();
			return name.includes(query) || url.includes(query);
		});
	});

	function getServerLabel(server: MCPServerSettingsEntry): string {
		return mcpStore.getServerLabel(server);
	}

	function isServerEnabledForChat(serverId: string): boolean {
		return conversationsStore.isMcpServerEnabledForChat(serverId);
	}

	async function toggleServerForChat(serverId: string) {
		await conversationsStore.toggleMcpServerForChat(serverId);
	}

	function handleMcpSubMenuOpen(open: boolean) {
		if (open) {
			mcpSearchQuery = '';
			mcpStore.runHealthChecksForServers(allMcpServers);
		}
	}

	function handleMcpSettingsClick() {
		onMcpSettingsClick?.();

		goto(`${hasMcpServers ? '' : '?add'}#/settings/mcp`);
	}
</script>

<DropdownMenu.Root>
	<DropdownMenu.Sub onOpenChange={handleMcpSubMenuOpen}>
		<DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
			<McpLogo class="h-4 w-4" />

			<span>MCP Servers</span>
		</DropdownMenu.SubTrigger>

		<DropdownMenu.SubContent class="w-72 pt-0">
			{#if hasMcpServers}
				<DropdownMenuSearchable
					placeholder="Search servers..."
					bind:searchValue={mcpSearchQuery}
					emptyMessage="No servers found"
					isEmpty={filteredMcpServers.length === 0}
				>
					<div class="max-h-64 overflow-y-auto">
						{#each filteredMcpServers as server (server.id)}
							{@const healthState = mcpStore.getHealthCheckState(server.id)}
							{@const hasError = healthState.status === HealthCheckStatus.ERROR}
							{@const isEnabledForChat = isServerEnabledForChat(server.id)}
							{@const displayName = getServerLabel(server)}
							{@const faviconUrl = mcpStore.getServerFavicon(server.id)}

							<button
								type="button"
								class="flex w-full items-center justify-between gap-2 rounded-sm px-2 py-2 text-left transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
								onclick={() => !hasError && toggleServerForChat(server.id)}
								disabled={hasError}
							>
								<div class="flex min-w-0 flex-1 items-center gap-2">
									<div class="min-w-0 flex-1">
										<McpServerIdentity
											{displayName}
											{faviconUrl}
											iconClass="h-4 w-4"
											iconRounded="rounded-sm"
											showVersion={false}
											nameClass="text-sm"
										/>
									</div>

									{#if hasError}
										<span
											class="shrink-0 rounded bg-destructive/15 px-1.5 py-0.5 text-xs text-destructive"
										>
											Error
										</span>
									{/if}
								</div>

								<Switch
									checked={isEnabledForChat}
									disabled={hasError}
									onclick={(e) => e.stopPropagation()}
									onCheckedChange={() => toggleServerForChat(server.id)}
								/>
							</button>
						{/each}
					</div>

					{#snippet footer()}
						<DropdownMenu.Item
							class="flex cursor-pointer items-center gap-2"
							onclick={handleMcpSettingsClick}
						>
							<Settings class="h-4 w-4" />

							<span>Manage MCP Servers</span>
						</DropdownMenu.Item>
					{/snippet}
				</DropdownMenuSearchable>
			{:else}
				<div class="px-2 py-3 text-center text-sm text-muted-foreground">
					No MCP servers configured
				</div>

				<DropdownMenu.Separator />

				<DropdownMenu.Item
					class="flex cursor-pointer items-center gap-2"
					onclick={handleMcpSettingsClick}
				>
					<Plus class="h-4 w-4" />

					<span>Add MCP Servers</span>
				</DropdownMenu.Item>
			{/if}
		</DropdownMenu.SubContent>
	</DropdownMenu.Sub>
</DropdownMenu.Root>
