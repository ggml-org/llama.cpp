<script lang="ts">
	import {
		McpConnectionLogs,
		McpServerCardActions,
		McpServerCardDeleteDialog,
		McpServerCardEditForm,
		McpServerCardHeader,
		McpServerCardToolsList,
		McpServerInfo
	} from '$lib/components/app/mcp';
	import * as Card from '$lib/components/ui/card';
	import { Skeleton } from '$lib/components/ui/skeleton';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { HealthCheckStatus } from '$lib/enums';
	import { mcpStore } from '$lib/stores';
	import type { HealthCheckState, MCPServerSettingsEntry } from '$lib/types';
	import { tick } from 'svelte';

	interface Props {
		server: MCPServerSettingsEntry;
		enabled?: boolean;
		onToggle: (enabled: boolean) => void;
		onUpdate: (updates: Partial<MCPServerSettingsEntry>) => void;
		onDelete: () => void;
	}

	let { enabled, onDelete, onToggle, onUpdate, server }: Props = $props();

	let healthState = $derived<HealthCheckState>(mcpStore.getHealthCheckState(server.id));
	let displayName = $derived(mcpStore.getServerLabel(server));
	let faviconUrl = $derived(mcpStore.getServerFavicon(server.id));
	let isIdle = $derived(healthState.status === HealthCheckStatus.IDLE);
	let isHealthChecking = $derived(healthState.status === HealthCheckStatus.CONNECTING);
	let isConnected = $derived(healthState.status === HealthCheckStatus.SUCCESS);
	let isError = $derived(healthState.status === HealthCheckStatus.ERROR);
	// Disabled servers stay IDLE (no startup health check), so the body
	// skeleton only applies while a check is running or expected to run.
	let showSkeleton = $derived(isHealthChecking || (isIdle && server.enabled));
	let errorMessage = $derived(
		healthState.status === HealthCheckStatus.ERROR ? healthState.message : undefined
	);
	let tools = $derived(healthState.status === HealthCheckStatus.SUCCESS ? healthState.tools : []);

	let connectionLogs = $derived(
		healthState.status === HealthCheckStatus.CONNECTING ||
			healthState.status === HealthCheckStatus.SUCCESS ||
			healthState.status === HealthCheckStatus.ERROR
			? healthState.logs
			: []
	);

	let successState = $derived(
		healthState.status === HealthCheckStatus.SUCCESS ? healthState : null
	);
	let serverInfo = $derived(successState?.serverInfo);
	let capabilities = $derived(successState?.capabilities);
	let transportType = $derived(successState?.transportType);
	let protocolVersion = $derived(successState?.protocolVersion);
	let connectionTimeMs = $derived(successState?.connectionTimeMs);
	let instructions = $derived(successState?.instructions);

	let isEditing = $derived(!server.url.trim());
	let showDeleteDialog = $state(false);
	let editFormRef: McpServerCardEditForm | null = $state(null);

	function handleHealthCheck() {
		mcpStore.runHealthCheck(server);
	}

	async function startEditing() {
		isEditing = true;
		await tick();
		editFormRef?.setInitialValues(
			server.url,
			server.headers || '',
			server.useProxy || false,
			displayName
		);
	}

	function cancelEditing() {
		if (server.url.trim()) {
			isEditing = false;
		} else {
			onDelete();
		}
	}

	function saveEditing(url: string, headers: string, useProxy: boolean, name?: string) {
		onUpdate({
			// undefined = prefill untouched, keep any existing custom name;
			// empty string = field cleared, back to the automatic label
			displayName: name === undefined ? server.displayName : name.trim() || undefined,
			headers: headers || undefined,
			url: url,
			useProxy: useProxy
		});
		isEditing = false;

		if (server.enabled && url) {
			setTimeout(() => mcpStore.runHealthCheck({ ...server, url, useProxy }), 100);
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
			onCancel={cancelEditing}
			onSave={saveEditing}
			serverId={server.id}
			serverLabel={displayName}
			serverUrl={server.url}
			serverUseProxy={server.useProxy}
		/>
	{:else}
		<McpServerCardHeader
			{capabilities}
			disabled={isError}
			{displayName}
			enabled={enabled ?? server.enabled}
			{faviconUrl}
			{onToggle}
			{serverInfo}
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
						<Skeleton class="{ICON_CLASS_DEFAULT} rounded" />

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
						<Skeleton class="{ICON_CLASS_DEFAULT} rounded" />

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
					<McpConnectionLogs {connectionTimeMs} logs={connectionLogs} />
				{/if}
			{/if}
		</div>

		<div class="mt-auto flex justify-between gap-4">
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
				onDelete={handleDeleteClick}
				onEdit={startEditing}
				onRefresh={handleHealthCheck}
			/>
		</div>
	{/if}
</Card.Root>

<McpServerCardDeleteDialog
	bind:open={showDeleteDialog}
	{displayName}
	onConfirm={onDelete}
	onOpenChange={(open) => (showDeleteDialog = open)}
/>
