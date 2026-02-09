<script lang="ts">
	import { Settings } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { Switch } from '$lib/components/ui/switch';
	import { DropdownMenuSearchable, McpActiveServersAvatars } from '$lib/components/app';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { getFaviconUrl } from '$lib/utils';
	import { HealthCheckStatus } from '$lib/enums';
	import type { MCPServerSettingsEntry } from '$lib/types';

	interface Props {
		class?: string;
		disabled?: boolean;
		onSettingsClick?: () => void;
	}

	let { class: className = '', disabled = false, onSettingsClick }: Props = $props();

	let searchQuery = $state('');
	let mcpServers = $derived(mcpStore.getServersSorted().filter((s) => s.enabled));
	let hasMcpServers = $derived(mcpServers.length > 0);
	let enabledMcpServersForChat = $derived(
		mcpServers.filter((s) => conversationsStore.isMcpServerEnabledForChat(s.id) && s.url.trim())
	);
	let healthyEnabledMcpServers = $derived(
		enabledMcpServersForChat.filter((s) => {
			const healthState = mcpStore.getHealthCheckState(s.id);
			return healthState.status !== HealthCheckStatus.ERROR;
		})
	);
	let hasEnabledMcpServers = $derived(enabledMcpServersForChat.length > 0);
	let mcpFavicons = $derived(
		healthyEnabledMcpServers
			.slice(0, 3)
			.map((s) => ({ id: s.id, url: getFaviconUrl(s.url) }))
			.filter((f) => f.url !== null)
	);
	let filteredMcpServers = $derived.by(() => {
		const query = searchQuery.toLowerCase().trim();
		if (query) {
			return mcpServers.filter((s) => {
				const name = getServerLabel(s).toLowerCase();
				const url = s.url.toLowerCase();
				return name.includes(query) || url.includes(query);
			});
		}
		return mcpServers;
	});

	function getServerLabel(server: MCPServerSettingsEntry): string {
		return mcpStore.getServerLabel(server);
	}

	function handleDropdownOpen(open: boolean) {
		if (open) {
			mcpStore.runHealthChecksForServers(mcpServers);
		}
	}

	function isServerEnabledForChat(serverId: string): boolean {
		return conversationsStore.isMcpServerEnabledForChat(serverId);
	}

	async function toggleServerForChat(serverId: string) {
		await conversationsStore.toggleMcpServerForChat(serverId);
	}
</script>

{#if hasMcpServers && hasEnabledMcpServers && mcpFavicons.length > 0}
	<DropdownMenu.Root
		onOpenChange={(open) => {
			if (!open) {
				searchQuery = '';
			}
			handleDropdownOpen(open);
		}}
	>
		<DropdownMenu.Trigger
			{disabled}
			onclick={(e) => {
				e.preventDefault();
				e.stopPropagation();
			}}
		>
			<button
				type="button"
				class="inline-flex cursor-pointer items-center rounded-sm py-1 disabled:cursor-not-allowed disabled:opacity-60"
				{disabled}
				aria-label="MCP Servers"
			>
				<McpActiveServersAvatars class={className} />
			</button>
		</DropdownMenu.Trigger>

		<DropdownMenu.Content align="start" class="w-72 pt-0">
			<DropdownMenuSearchable
				bind:searchValue={searchQuery}
				placeholder="Search servers..."
				emptyMessage="No servers found"
				isEmpty={filteredMcpServers.length === 0}
			>
				<div class="max-h-64 overflow-y-auto">
					{#each filteredMcpServers as server (server.id)}
						{@const healthState = mcpStore.getHealthCheckState(server.id)}
						{@const hasError = healthState.status === HealthCheckStatus.ERROR}
						{@const isEnabledForChat = isServerEnabledForChat(server.id)}

						<button
							type="button"
							class="flex w-full items-center justify-between gap-2 px-2 py-2 text-left transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
							onclick={() => !hasError && toggleServerForChat(server.id)}
							disabled={hasError}
						>
							<div class="flex min-w-0 flex-1 items-center gap-2">
								{#if getFaviconUrl(server.url)}
									<img
										src={getFaviconUrl(server.url)}
										alt=""
										class="h-4 w-4 shrink-0 rounded-sm"
										onerror={(e) => {
											(e.currentTarget as HTMLImageElement).style.display = 'none';
										}}
									/>
								{/if}

								<span class="truncate text-sm">{getServerLabel(server)}</span>

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
								onclick={(e: MouseEvent) => e.stopPropagation()}
								onCheckedChange={() => toggleServerForChat(server.id)}
							/>
						</button>
					{/each}
				</div>

				{#snippet footer()}
					<DropdownMenu.Item
						class="flex cursor-pointer items-center gap-2"
						onclick={onSettingsClick}
					>
						<Settings class="h-4 w-4" />

						<span>Manage MCP Servers</span>
					</DropdownMenu.Item>
				{/snippet}
			</DropdownMenuSearchable>
		</DropdownMenu.Content>
	</DropdownMenu.Root>
{/if}
