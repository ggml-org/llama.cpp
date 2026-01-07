<script lang="ts">
	import { onMount } from 'svelte';
	import {
		Trash2,
		RefreshCw,
		Cable,
		ChevronDown,
		ChevronRight,
		Pencil,
		X,
		ExternalLink
	} from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Switch } from '$lib/components/ui/switch';
	import * as Card from '$lib/components/ui/card';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import McpServerForm from '$lib/components/app/mcp/McpServerForm.svelte';
	import type { MCPServerSettingsEntry } from '$lib/types/mcp';
	import {
		mcpGetHealthCheckState,
		mcpHasHealthCheck,
		mcpRunHealthCheck,
		type HealthCheckState
	} from '$lib/stores/mcp.svelte';
	import { Badge } from '$lib/components/ui/badge';

	interface Props {
		server: MCPServerSettingsEntry;
		displayName: string;
		faviconUrl: string | null;
		onToggle: (enabled: boolean) => void;
		onUpdate: (updates: Partial<MCPServerSettingsEntry>) => void;
		onDelete: () => void;
	}

	let { server, displayName, faviconUrl, onToggle, onUpdate, onDelete }: Props = $props();

	let healthState = $derived<HealthCheckState>(mcpGetHealthCheckState(server.id));
	let isHealthChecking = $derived(healthState.status === 'loading');
	let isConnected = $derived(healthState.status === 'success');
	let isError = $derived(healthState.status === 'error');
	let errorMessage = $derived(healthState.status === 'error' ? healthState.message : undefined);
	let tools = $derived(healthState.status === 'success' ? healthState.tools : []);
	let toolsCount = $derived(tools.length);

	let isExpanded = $state(false);

	let showDeleteDialog = $state(false);

	let isEditing = $state(!server.url.trim());
	let editUrl = $state(server.url);
	let editHeaders = $state(server.headers || '');

	let urlError = $derived.by(() => {
		if (!editUrl.trim()) return 'URL is required';
		try {
			new URL(editUrl);
			return null;
		} catch {
			return 'Invalid URL format';
		}
	});

	let canSave = $derived(!urlError);

	onMount(() => {
		if (!mcpHasHealthCheck(server.id) && server.enabled && server.url.trim()) {
			mcpRunHealthCheck(server);
		}
	});

	function handleHealthCheck() {
		mcpRunHealthCheck(server);
	}

	function startEditing() {
		editUrl = server.url;
		editHeaders = server.headers || '';
		isEditing = true;
	}

	function cancelEditing() {
		if (server.url.trim()) {
			editUrl = server.url;
			editHeaders = server.headers || '';
			isEditing = false;
		} else {
			onDelete();
		}
	}

	function saveEditing() {
		if (!canSave) return;
		onUpdate({
			url: editUrl.trim(),
			headers: editHeaders.trim() || undefined
		});
		isEditing = false;

		if (server.enabled && editUrl.trim()) {
			setTimeout(() => mcpRunHealthCheck({ ...server, url: editUrl.trim() }), 100);
		}
	}
</script>

<Card.Root class="!gap-4 bg-muted/30 p-4">
	{#if isEditing}
		<div class="space-y-4">
			<div class="flex items-center justify-between">
				<p class="font-medium">Configure Server</p>
				<Button
					variant="ghost"
					size="icon"
					class="h-7 w-7"
					onclick={cancelEditing}
					aria-label="Cancel"
				>
					<X class="h-3.5 w-3.5" />
				</Button>
			</div>

			<McpServerForm
				url={editUrl}
				headers={editHeaders}
				onUrlChange={(v) => (editUrl = v)}
				onHeadersChange={(v) => (editHeaders = v)}
				urlError={editUrl ? urlError : null}
				id={server.id}
			/>

			<div class="flex items-center justify-end">
				<Button
					variant="default"
					size="sm"
					onclick={saveEditing}
					disabled={!canSave}
					aria-label="Save"
				>
					{server.url.trim() ? 'Update' : 'Add'}
				</Button>
			</div>
		</div>
	{:else}
		<div class="flex items-center justify-between gap-3">
			<div class="flex min-w-0 flex-1 items-center gap-2">
				{#if faviconUrl}
					<img
						src={faviconUrl}
						alt=""
						class="h-5 w-5 shrink-0 rounded"
						onerror={(e) => {
							(e.currentTarget as HTMLImageElement).style.display = 'none';
						}}
					/>
				{:else}
					<Cable class="h-5 w-5 shrink-0 text-muted-foreground" />
				{/if}
				<p class="truncate font-medium">{displayName}</p>
				{#if server.url}
					<a
						href={server.url}
						target="_blank"
						rel="noopener noreferrer"
						class="shrink-0 text-muted-foreground hover:text-foreground"
						aria-label="Open server URL"
					>
						<ExternalLink class="h-3.5 w-3.5" />
					</a>
				{/if}
				{#if isHealthChecking}
					<span class="shrink-0 text-xs text-muted-foreground">Checking...</span>
				{:else if isConnected}
					<span
						class="shrink-0 rounded bg-green-500/15 px-1.5 py-0.5 text-xs text-green-600 dark:text-green-500"
						>Connected</span
					>
				{:else if isError}
					<span class="shrink-0 rounded bg-destructive/15 px-1.5 py-0.5 text-xs text-destructive"
						>Error</span
					>
				{/if}
			</div>

			<div class="flex shrink-0 items-center">
				<Switch checked={server.enabled} onCheckedChange={onToggle} />
			</div>
		</div>

		{#if isError && errorMessage}
			<p class="mt-3 text-xs text-destructive">{errorMessage}</p>
		{/if}

		{#if tools.length === 0 && server.url.trim()}
			<div class="mt-3 flex items-center justify-end gap-1">
				<Button
					variant="ghost"
					size="icon"
					class="h-7 w-7"
					onclick={startEditing}
					aria-label="Edit"
				>
					<Pencil class="h-3.5 w-3.5" />
				</Button>
				<Button
					variant="ghost"
					size="icon"
					class="h-7 w-7"
					onclick={handleHealthCheck}
					disabled={isHealthChecking}
					aria-label="Refresh"
				>
					<RefreshCw class="h-3.5 w-3.5" />
				</Button>
				<Button
					variant="ghost"
					size="icon"
					class="hover:text-destructive-foreground h-7 w-7 text-destructive hover:bg-destructive"
					onclick={() => (showDeleteDialog = true)}
					aria-label="Delete"
				>
					<Trash2 class="h-3.5 w-3.5" />
				</Button>
			</div>
		{/if}

		{#if tools.length > 0}
			<Collapsible.Root bind:open={isExpanded}>
				<div class="flex items-center justify-between gap-3">
					<Collapsible.Trigger
						class="flex flex-1 items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
					>
						{#if isExpanded}
							<ChevronDown class="h-3.5 w-3.5" />
						{:else}
							<ChevronRight class="h-3.5 w-3.5" />
						{/if}
						<span>{toolsCount} tools available · Show details</span>
					</Collapsible.Trigger>
					<div class="flex shrink-0 items-center gap-1">
						<Button
							variant="ghost"
							size="icon"
							class="h-7 w-7"
							onclick={startEditing}
							aria-label="Edit"
						>
							<Pencil class="h-3.5 w-3.5" />
						</Button>
						<Button
							variant="ghost"
							size="icon"
							class="h-7 w-7"
							onclick={handleHealthCheck}
							disabled={isHealthChecking}
							aria-label="Refresh"
						>
							<RefreshCw class="h-3.5 w-3.5" />
						</Button>
						<Button
							variant="ghost"
							size="icon"
							class="hover:text-destructive-foreground h-7 w-7 text-destructive hover:bg-destructive"
							onclick={() => (showDeleteDialog = true)}
							aria-label="Delete"
						>
							<Trash2 class="h-3.5 w-3.5" />
						</Button>
					</div>
				</div>
				<Collapsible.Content class="mt-2">
					<div class="max-h-64 space-y-3 overflow-y-auto">
						{#each tools as tool (tool.name)}
							<div>
								<Badge variant="secondary">{tool.name}</Badge>
								{#if tool.description}
									<p class="mt-1 text-xs text-muted-foreground">{tool.description}</p>
								{/if}
							</div>
						{/each}
					</div>
				</Collapsible.Content>
			</Collapsible.Root>
		{/if}
	{/if}
</Card.Root>

<AlertDialog.Root bind:open={showDeleteDialog}>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title>Delete Server</AlertDialog.Title>
			<AlertDialog.Description>
				Are you sure you want to delete <strong>{displayName}</strong>? This action cannot be
				undone.
			</AlertDialog.Description>
		</AlertDialog.Header>
		<AlertDialog.Footer>
			<AlertDialog.Cancel>Cancel</AlertDialog.Cancel>
			<AlertDialog.Action
				class="text-destructive-foreground bg-destructive hover:bg-destructive/90"
				onclick={onDelete}
			>
				Delete
			</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
