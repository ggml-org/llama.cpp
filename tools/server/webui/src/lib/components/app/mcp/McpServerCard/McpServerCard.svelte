<script lang="ts">
	import { onMount } from 'svelte';
	import * as Card from '$lib/components/ui/card';
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
	import Badge from '$lib/components/ui/badge/badge.svelte';

	interface Props {
		server: MCPServerSettingsEntry;
		displayName: string;
		faviconUrl: string | null;
		onToggle: (enabled: boolean) => void;
		onUpdate: (updates: Partial<MCPServerSettingsEntry>) => void;
		onDelete: () => void;
	}

	let { server, displayName, faviconUrl, onToggle, onUpdate, onDelete }: Props = $props();

	let healthState = $derived<HealthCheckState>(mcpStore.getHealthCheckState(server.id));
	let isHealthChecking = $derived(healthState.status === HealthCheckStatus.Connecting);
	let isConnected = $derived(healthState.status === HealthCheckStatus.Success);
	let isError = $derived(healthState.status === HealthCheckStatus.Error);
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

	const transportLabels: Record<string, string> = {
		websocket: 'WebSocket',
		streamable_http: 'HTTP',
		sse: 'SSE'
	};

	onMount(() => {
		if (!mcpStore.hasHealthCheck(server.id) && server.enabled && server.url.trim()) {
			mcpClient.runHealthCheck(server);
		}
	});

	function handleHealthCheck() {
		mcpClient.runHealthCheck(server);
	}

	function startEditing() {
		editFormRef?.setInitialValues(server.url, server.headers || '');
		isEditing = true;
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
			enabled={server.enabled}
			{onToggle}
			{serverInfo}
			{capabilities}
		/>

		{#if isError && errorMessage}
			<p class="mt-2 text-xs text-destructive">{errorMessage}</p>
		{/if}

		{#if isConnected && serverInfo?.description}
			<p class="mt-3 line-clamp-2 text-xs text-muted-foreground">
				{serverInfo.description}
			</p>
		{/if}

		<div class="mt-2 grid gap-3">
			{#if isConnected && instructions}
				<McpServerInfo {instructions} class="mt-3" />
			{/if}

			{#if tools.length > 0}
				<McpServerCardToolsList {tools} />
			{/if}

			{#if connectionLogs.length > 0}
				<McpConnectionLogs logs={connectionLogs} {connectionTimeMs} />
			{/if}
		</div>

		<div class="mt-4 flex justify-between gap-4">
			{#if transportType || protocolVersion}
				<div class="flex flex-wrap items-center gap-1">
					{#if transportType}
						<Badge variant="outline" class="h-5 gap-1 px-1.5 text-[10px]">
							{transportLabels[transportType] || transportType}
						</Badge>
					{/if}

					{#if protocolVersion}
						<Badge variant="outline" class="h-5 gap-1 px-1.5 text-[10px]">
							MCP {protocolVersion}
						</Badge>
					{/if}
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
