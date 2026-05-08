<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { HealthCheckStatus } from '$lib/enums';
	import { MAX_DISPLAYED_MCP_AVATARS } from '$lib/constants';
	import McpLogo from './McpLogo.svelte';

	interface Props {
		class?: string;
		onclick?: () => void;
	}

	let { class: className = '', onclick }: Props = $props();

	let mcpServers = $derived(mcpStore.getServersSorted().filter((s) => s.enabled));
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
	let extraServersCount = $derived(
		Math.max(0, healthyEnabledMcpServers.length - MAX_DISPLAYED_MCP_AVATARS)
	);
	let mcpFavicons = $derived(
		healthyEnabledMcpServers
			.slice(0, MAX_DISPLAYED_MCP_AVATARS)
			.map((s) => ({
				id: s.id,
				name: mcpStore.getServerDisplayName(s.id),
				url: mcpStore.getServerFavicon(s.id)
			}))
			.filter((f) => f.url !== null)
	);
</script>

{#if !hasEnabledMcpServers}
	<button
		class={[
			'inline-flex cursor-pointer items-center gap-0.75 opacity-70 transition-opacity hover:opacity-100',
			className,
			'opacity-50 hover:opacity-100'
		]}
		{onclick}
	>
		<Tooltip.Root>
			<Tooltip.Trigger>
				<McpLogo class="h-4 w-4" />
			</Tooltip.Trigger>

			<Tooltip.Content>
				<p>MCP Servers</p>
			</Tooltip.Content>
		</Tooltip.Root>
	</button>
{:else if mcpFavicons.length > 0}
	<button class={['inline-flex items-center gap-0.75', className]} {onclick}>
		<div class="flex -space-x-1">
			{#each mcpFavicons as favicon (favicon.id)}
				<Tooltip.Root>
					<Tooltip.Trigger>
						<div class="box-shadow-lg overflow-hidden rounded-full bg-muted ring-1 ring-muted">
							<img
								src={favicon.url}
								alt=""
								class="h-4 w-4"
								onerror={(e) => {
									(e.currentTarget as HTMLImageElement).style.display = 'none';
								}}
							/>
						</div>
					</Tooltip.Trigger>
					<Tooltip.Content>
						<p>{favicon.name}</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/each}
		</div>

		{#if extraServersCount > 0}
			<span class="text-xs text-muted-foreground">+{extraServersCount}</span>
		{/if}
	</button>
{/if}
