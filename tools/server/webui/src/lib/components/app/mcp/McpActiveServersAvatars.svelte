<script lang="ts">
	import { cn } from '$lib/components/ui/utils';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { getFaviconUrl } from '$lib/utils/mcp';
	import { HealthCheckStatus } from '$lib/enums';

	interface Props {
		class?: string;
		disabled?: boolean;
		onSettingsClick?: () => void;
	}

	let { class: className = '', disabled = false, onSettingsClick }: Props = $props();

	let mcpServers = $derived(mcpStore.getServersSorted().filter((s) => s.enabled));
	let enabledMcpServersForChat = $derived(
		mcpServers.filter((s) => conversationsStore.isMcpServerEnabledForChat(s.id) && s.url.trim())
	);
	let healthyEnabledMcpServers = $derived(
		enabledMcpServersForChat.filter((s) => {
			const healthState = mcpStore.getHealthCheckState(s.id);
			return healthState.status !== HealthCheckStatus.Error;
		})
	);
	let hasEnabledMcpServers = $derived(enabledMcpServersForChat.length > 0);
	let extraServersCount = $derived(Math.max(0, healthyEnabledMcpServers.length - 3));
	let mcpFavicons = $derived(
		healthyEnabledMcpServers
			.slice(0, 3)
			.map((s) => ({ id: s.id, url: getFaviconUrl(s.url) }))
			.filter((f) => f.url !== null)
	);
</script>

{#if hasEnabledMcpServers && mcpFavicons.length > 0}
	<button
		type="button"
		class={cn(
			'inline-flex items-center gap-1.5 rounded-sm py-1',
			disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer',
			className
		)}
		onclick={onSettingsClick}
		{disabled}
		aria-label="MCP Servers"
	>
		<div class="flex -space-x-1">
			{#each mcpFavicons as favicon (favicon.id)}
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
			{/each}
		</div>

		{#if extraServersCount > 0}
			<span class="text-xs text-muted-foreground">+{extraServersCount}</span>
		{/if}
	</button>
{/if}
