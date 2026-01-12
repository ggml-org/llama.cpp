<script lang="ts">
	import { onMount } from 'svelte';
	import * as Card from '$lib/components/ui/card';
	import type { MCPServerSettingsEntry } from '$lib/types/mcp';
	import { mcpStore, type HealthCheckState } from '$lib/stores/mcp.svelte';
	import { mcpClient } from '$lib/clients/mcp.client';
	import McpServerCardHeader from './McpServerCardHeader.svelte';
	import McpServerCardActions from './McpServerCardActions.svelte';
	import McpServerCardToolsList from './McpServerCardToolsList.svelte';
	import McpServerCardEditForm from './McpServerCardEditForm.svelte';
	import McpServerCardDeleteDialog from './McpServerCardDeleteDialog.svelte';

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
	let isHealthChecking = $derived(healthState.status === 'loading');
	let isConnected = $derived(healthState.status === 'success');
	let isError = $derived(healthState.status === 'error');
	let errorMessage = $derived(healthState.status === 'error' ? healthState.message : undefined);
	let tools = $derived(healthState.status === 'success' ? healthState.tools : []);

	let isEditing = $state(!server.url.trim());
	let showDeleteDialog = $state(false);
	let editFormRef: McpServerCardEditForm | null = $state(null);

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

<Card.Root class="!gap-4 bg-muted/30 p-4">
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
			serverUrl={server.url}
			enabled={server.enabled}
			{isHealthChecking}
			{isConnected}
			{isError}
			{onToggle}
		/>

		{#if isError && errorMessage}
			<p class="mt-3 text-xs text-destructive">{errorMessage}</p>
		{/if}

		{#if tools.length === 0 && server.url.trim()}
			<div class="mt-3 flex items-center justify-end gap-1">
				<McpServerCardActions
					{isHealthChecking}
					onEdit={startEditing}
					onRefresh={handleHealthCheck}
					onDelete={handleDeleteClick}
				/>
			</div>
		{/if}

		{#if tools.length > 0}
			<McpServerCardToolsList
				{tools}
				{isHealthChecking}
				onEdit={startEditing}
				onRefresh={handleHealthCheck}
				onDelete={handleDeleteClick}
			/>
		{/if}
	{/if}
</Card.Root>

<McpServerCardDeleteDialog
	bind:open={showDeleteDialog}
	{displayName}
	onOpenChange={(open) => (showDeleteDialog = open)}
	onConfirm={onDelete}
/>
