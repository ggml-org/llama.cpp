<script lang="ts">
	import { tick } from 'svelte';
	import * as Card from '$lib/components/ui/card';
	import { Skeleton } from '$lib/components/ui/skeleton';
	import type { MCPServerSettingsEntry, HealthCheckState } from '$lib/types';
	import { HealthCheckStatus } from '$lib/enums';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { mcpClient } from '$lib/clients/mcp.client';
	import McpServerCardHeader from './McpServerCardHeader.svelte';
	import McpServerCardActions from './McpServerCardActions.svelte';
	import McpServerCardToolsList from './McpServerCardToolsList.svelte';
	import McpServerCardEditForm from './McpServerCardEditForm.svelte';
	import McpServerCardDeleteDialog from './McpServerCardDeleteDialog.svelte';
	import McpServerInfo from './McpServerInfo.svelte';
	import McpConnectionLogs from './McpConnectionLogs.svelte';

	interface Props {
		server: MCPServerSettingsEntry;
		faviconUrl: string | null;
		enabled?: boolean;
		onToggle: (enabled: boolean) => void;
		onUpdate: (updates: Partial<MCPServerSettingsEntry>) => void;
		onDelete: () => void;
	}

	let { server, faviconUrl, enabled, onToggle, onUpdate, onDelete }: Props = $props();

	let healthState = $derived<HealthCheckState>(mcpStore.getHealthCheckState(server.id));
	let displayName = $derived(mcpStore.getServerLabel(server));
	let isIdle = $derived(healthState.status === HealthCheckStatus.Idle);
	let isHealthChecking = $derived(healthState.status === HealthCheckStatus.Connecting);
	let isConnected = $derived(healthState.status === HealthCheckStatus.Success);
	let isError = $derived(healthState.status === HealthCheckStatus.Error);
	let showSkeleton = $derived(isIdle || isHealthChecking);
	let errorMessage = $derived(
		healthState.status === HealthCheckStatus.Error ? healthState.message : undefined
	);
	let tools = $derived(healthState.status === HealthCheckStatus.Success ? healthState.tools : []);

	let connectionLogs = $derived(
		healthState.status === HealthCheckStatus.Connecting ||
			healthState.status === HealthCheckStatus.Success ||
			healthState.status === HealthCheckStatus.Error
			? healthState.logs
			: []
	);

	let serverInfo = $derived(
		healthState.status === HealthCheckStatus.Success ? healthState.serverInfo : undefined
	);
	let capabilities = $derived(
		healthState.status === HealthCheckStatus.Success ? healthState.capabilities : undefined
	);
	let transportType = $derived(
		healthState.status === HealthCheckStatus.Success ? healthState.transportType : undefined
	);
	let protocolVersion = $derived(
		healthState.status === HealthCheckStatus.Success ? healthState.protocolVersion : undefined
	);
	let connectionTimeMs = $derived(
		healthState.status === HealthCheckStatus.Success ? healthState.connectionTimeMs : undefined
	);
	let instructions = $derived(
		healthState.status === HealthCheckStatus.Success ? healthState.instructions : undefined
	);

	let isEditing = $state(!server.url.trim());
	let showDeleteDialog = $state(false);
	let editFormRef: McpServerCardEditForm | null = $state(null);

	function handleHealthCheck() {
		mcpClient.runHealthCheck(server);
	}

	async function startEditing() {
		isEditing = true;
		await tick();
		editFormRef?.setInitialValues(server.url, server.headers || '');
	}

	function cancelEditing() {
		if (server.url.trim()) {
			isEditing = false;
		} else {
			onDelete();
		}
	}

	function saveEditing(url: string, headers: string) {
		onUpdate({
			url: url,
			headers: headers || undefined
		});
		isEditing = false;

		if (server.enabled && url) {
			setTimeout(() => mcpClient.runHealthCheck({ ...server, url }), 100);
		}
	}

	function handleDeleteClick() {
		showDeleteDialog = true;
	}
</script>

<Card.Root class="!gap-3 bg-muted/30 p-4">
	{#if isEditing}
		<McpServerCardEditForm
			bind:this={editFormRef}
			serverId={server.id}
			serverUrl={server.url}
			onSave={saveEditing}
			onCancel={cancelEditing}
		/>
	{:else}
		<McpServerCardHeader
			{displayName}
			{faviconUrl}
			enabled={enabled ?? server.enabled}
			{onToggle}
			{serverInfo}
			{capabilities}
			{transportType}
		/>

		{#if isError && errorMessage}
			<p class="text-xs text-destructive">{errorMessage}</p>
		{/if}

		{#if isConnected && serverInfo?.description}
			<p class="line-clamp-2 text-xs text-muted-foreground">
				{serverInfo.description}
			</p>
		{/if}

		<div class="grid gap-3">
			{#if showSkeleton}
				<div class="space-y-2">
					<div class="flex items-center gap-2">
						<Skeleton class="h-4 w-4 rounded" />
						<Skeleton class="h-3 w-24" />
					</div>
					<div class="flex flex-wrap gap-1.5">
						<Skeleton class="h-5 w-16 rounded-full" />
						<Skeleton class="h-5 w-20 rounded-full" />
						<Skeleton class="h-5 w-14 rounded-full" />
					</div>
				</div>

				<div class="space-y-1.5">
					<div class="flex items-center gap-2">
						<Skeleton class="h-4 w-4 rounded" />
						<Skeleton class="h-3 w-32" />
					</div>
				</div>
			{:else}
				{#if isConnected && instructions}
					<McpServerInfo {instructions} />
				{/if}

				{#if tools.length > 0}
					<McpServerCardToolsList {tools} />
				{/if}

				{#if connectionLogs.length > 0}
					<McpConnectionLogs logs={connectionLogs} {connectionTimeMs} />
				{/if}
			{/if}
		</div>

		<div class="flex justify-between gap-4">
			{#if showSkeleton}
				<Skeleton class="h-3 w-28" />
			{:else if protocolVersion}
				<div class="flex flex-wrap items-center gap-1">
					<span class="text-[10px] text-muted-foreground">
						Protocol version: {protocolVersion}
					</span>
				</div>
			{/if}

			<McpServerCardActions
				{isHealthChecking}
				onEdit={startEditing}
				onRefresh={handleHealthCheck}
				onDelete={handleDeleteClick}
			/>
		</div>
	{/if}
</Card.Root>

<McpServerCardDeleteDialog
	bind:open={showDeleteDialog}
	{displayName}
	onOpenChange={(open) => (showDeleteDialog = open)}
	onConfirm={onDelete}
/>
