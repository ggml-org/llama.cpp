<script lang="ts">
	import { onMount } from 'svelte';
	import { Square, Settings, ChevronDown, Search } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { Switch } from '$lib/components/ui/switch';
	import { cn } from '$lib/components/ui/utils';
	import {
		ChatFormActionFileAttachments,
		ChatFormActionRecord,
		ChatFormActionSubmit,
		DialogMcpServersSettings,
		ModelsSelector
	} from '$lib/components/app';
	import McpLogo from '$lib/components/app/misc/McpLogo.svelte';
	import { FileTypeCategory } from '$lib/enums';
	import { getFileTypeCategory } from '$lib/utils';
	import { config } from '$lib/stores/settings.svelte';
	import { modelsStore, modelOptions, selectedModelId } from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import {
		activeMessages,
		usedModalities,
		conversationsStore
	} from '$lib/stores/conversations.svelte';
	import { useModelChangeValidation } from '$lib/hooks/use-model-change-validation.svelte';
	import { parseMcpServerSettings, parseMcpServerUsageStats } from '$lib/config/mcp';
	import type { MCPServerSettingsEntry } from '$lib/types/mcp';
	import {
		mcpGetHealthCheckState,
		mcpHasHealthCheck,
		mcpRunHealthCheck
	} from '$lib/stores/mcp.svelte';

	interface Props {
		canSend?: boolean;
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		isRecording?: boolean;
		hasText?: boolean;
		uploadedFiles?: ChatUploadedFile[];
		onFileUpload?: (fileType?: FileTypeCategory) => void;
		onMicClick?: () => void;
		onStop?: () => void;
	}

	let {
		canSend = false,
		class: className = '',
		disabled = false,
		isLoading = false,
		isRecording = false,
		hasText = false,
		uploadedFiles = [],
		onFileUpload,
		onMicClick,
		onStop
	}: Props = $props();

	let currentConfig = $derived(config());
	let isRouter = $derived(isRouterMode());

	let conversationModel = $derived(
		chatStore.getConversationModel(activeMessages() as DatabaseMessage[])
	);

	let previousConversationModel: string | null = null;

	$effect(() => {
		if (conversationModel && conversationModel !== previousConversationModel) {
			previousConversationModel = conversationModel;
			modelsStore.selectModelByName(conversationModel);
		}
	});

	let activeModelId = $derived.by(() => {
		const options = modelOptions();

		if (!isRouter) {
			return options.length > 0 ? options[0].model : null;
		}

		const selectedId = selectedModelId();
		if (selectedId) {
			const model = options.find((m) => m.id === selectedId);
			if (model) return model.model;
		}

		if (conversationModel) {
			const model = options.find((m) => m.model === conversationModel);
			if (model) return model.model;
		}

		return null;
	});

	let modelPropsVersion = $state(0); // Used to trigger reactivity after fetch

	$effect(() => {
		if (activeModelId) {
			const cached = modelsStore.getModelProps(activeModelId);

			if (!cached) {
				modelsStore.fetchModelProps(activeModelId).then(() => {
					modelPropsVersion++;
				});
			}
		}
	});

	let hasAudioModality = $derived.by(() => {
		if (activeModelId) {
			void modelPropsVersion;

			return modelsStore.modelSupportsAudio(activeModelId);
		}

		return false;
	});

	let hasVisionModality = $derived.by(() => {
		if (activeModelId) {
			void modelPropsVersion;

			return modelsStore.modelSupportsVision(activeModelId);
		}

		return false;
	});

	let hasAudioAttachments = $derived(
		uploadedFiles.some((file) => getFileTypeCategory(file.type) === FileTypeCategory.AUDIO)
	);
	let shouldShowRecordButton = $derived(
		hasAudioModality && !hasText && !hasAudioAttachments && currentConfig.autoMicOnEmpty
	);

	let hasModelSelected = $derived(!isRouter || !!conversationModel || !!selectedModelId());

	let isSelectedModelInCache = $derived.by(() => {
		if (!isRouter) return true;

		if (conversationModel) {
			return modelOptions().some((option) => option.model === conversationModel);
		}

		const currentModelId = selectedModelId();
		if (!currentModelId) return false;

		return modelOptions().some((option) => option.id === currentModelId);
	});

	let submitTooltip = $derived.by(() => {
		if (!hasModelSelected) {
			return 'Please select a model first';
		}

		if (!isSelectedModelInCache) {
			return 'Selected model is not available, please select another';
		}

		return '';
	});

	let selectorModelRef: ModelsSelector | undefined = $state(undefined);

	export function openModelSelector() {
		selectorModelRef?.open();
	}

	const { handleModelChange } = useModelChangeValidation({
		getRequiredModalities: () => usedModalities(),
		onValidationFailure: async (previousModelId) => {
			if (previousModelId) {
				await modelsStore.selectModelById(previousModelId);
			}
		}
	});

	let showMcpDialog = $state(false);
	let mcpSearchQuery = $state('');

	// MCP servers state
	let mcpServers = $derived<MCPServerSettingsEntry[]>(
		parseMcpServerSettings(currentConfig.mcpServers)
	);

	// Usage stats for sorting by popularity
	let mcpUsageStats = $derived(parseMcpServerUsageStats(currentConfig.mcpServerUsageStats));

	// Get usage count for a server
	function getServerUsageCount(serverId: string): number {
		return mcpUsageStats[serverId] || 0;
	}

	// Helper to check if server is enabled for current chat (per-chat override or global)
	function isServerEnabledForChat(server: MCPServerSettingsEntry): boolean {
		return conversationsStore.isMcpServerEnabledForChat(server.id, server.enabled);
	}

	// Helper to check if server has per-chat override
	function hasPerChatOverride(serverId: string): boolean {
		return conversationsStore.getMcpServerOverride(serverId) !== undefined;
	}

	// Servers enabled for current chat (considering per-chat overrides)
	let enabledMcpServersForChat = $derived(
		mcpServers.filter((s) => isServerEnabledForChat(s) && s.url.trim())
	);
	// Filter out servers with health check errors
	let healthyEnabledMcpServers = $derived(
		enabledMcpServersForChat.filter((s) => {
			const healthState = mcpGetHealthCheckState(s.id);
			return healthState.status !== 'error';
		})
	);
	let hasEnabledMcpServers = $derived(enabledMcpServersForChat.length > 0);
	let hasMcpServers = $derived(mcpServers.length > 0);

	// Sort servers: globally enabled first (by popularity), then rest (by popularity)
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

	// Filtered and limited servers for dropdown display
	let displayedMcpServers = $derived(() => {
		const query = mcpSearchQuery.toLowerCase().trim();
		if (query) {
			// When searching, show all matching results
			return sortedMcpServers.filter((s) => {
				const name = getServerDisplayName(s).toLowerCase();
				const url = s.url.toLowerCase();
				return name.includes(query) || url.includes(query);
			});
		}
		// When not searching, show max 4 items
		return sortedMcpServers.slice(0, 4);
	});

	// Count of extra servers beyond the 3 shown as favicons (excluding error servers)
	let extraServersCount = $derived(Math.max(0, healthyEnabledMcpServers.length - 3));

	// Toggle server enabled state for current chat (per-chat override only)
	// Global state is only changed from MCP Settings Dialog
	async function toggleServerForChat(serverId: string, globalEnabled: boolean) {
		await conversationsStore.toggleMcpServerForChat(serverId, globalEnabled);
	}

	// Get display name for server
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

	// Get favicon URLs for enabled servers (max 3)
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

	// All servers with valid URLs (for health checks - check all regardless of enabled state)
	let serversWithUrls = $derived(mcpServers.filter((s) => s.url.trim()));

	// Run health checks on mount for ALL servers with URLs (to show metadata in dropdown)
	onMount(() => {
		for (const server of serversWithUrls) {
			if (!mcpHasHealthCheck(server.id)) {
				mcpRunHealthCheck(server);
			}
		}
	});
</script>

<div class="flex w-full items-center gap-3 {className}" style="container-type: inline-size">
	<div class="mr-auto flex items-center gap-1.5">
		<ChatFormActionFileAttachments
			{disabled}
			{hasAudioModality}
			{hasVisionModality}
			showMcpOption={!hasMcpServers}
			onMcpClick={() => (showMcpDialog = true)}
			{onFileUpload}
		/>

		{#if hasMcpServers}
			<DropdownMenu.Root>
				<DropdownMenu.Trigger {disabled}>
					<button
						type="button"
						class={cn(
							'inline-flex cursor-pointer items-center rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60',
							hasEnabledMcpServers ? 'text-foreground' : 'text-muted-foreground'
						)}
						{disabled}
						aria-label="MCP Servers"
					>
						<McpLogo style="width: 0.875rem; height: 0.875rem;" />

						<span class="mx-1.5 font-medium"> MCP </span>

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

							{#if hasEnabledMcpServers && extraServersCount > 0}
								<span class="ml-1 text-muted-foreground">+{extraServersCount}</span>
							{/if}
						{/if}

						<ChevronDown class="h-3 w-3.5" />
					</button>
				</DropdownMenu.Trigger>

				<DropdownMenu.Content align="start" class="w-72">
					<!-- Search Input -->
					<div class="px-2 pb-2">
						<div class="relative">
							<Search
								class="absolute top-1/2 left-2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground"
							/>
							<input
								type="text"
								placeholder="Search servers..."
								bind:value={mcpSearchQuery}
								class="h-8 w-full rounded-md border border-input bg-background pr-3 pl-8 text-sm placeholder:text-muted-foreground focus:ring-1 focus:ring-ring focus:outline-none"
							/>
						</div>
					</div>

					<!-- Servers (sorted by popularity then alphabetically, max 4 when no search) -->
					<div class="max-h-48 overflow-y-auto">
						{#each displayedMcpServers() as server (server.id)}
							{@const healthState = mcpGetHealthCheckState(server.id)}
							{@const hasError = healthState.status === 'error'}
							{@const isEnabledForChat = isServerEnabledForChat(server)}
							{@const hasOverride = hasPerChatOverride(server.id)}
							<div class="flex items-center justify-between gap-2 px-2 py-1.5">
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
										<span
											class="shrink-0 rounded bg-destructive/15 px-1.5 py-0.5 text-xs text-destructive"
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
						{#if displayedMcpServers().length === 0}
							<div class="px-2 py-3 text-center text-sm text-muted-foreground">
								No servers found
							</div>
						{/if}
					</div>
					<DropdownMenu.Separator />
					<DropdownMenu.Item
						class="flex cursor-pointer items-center gap-2"
						onclick={() => (showMcpDialog = true)}
					>
						<Settings class="h-4 w-4" />
						<span>Manage MCP Servers</span>
					</DropdownMenu.Item>
				</DropdownMenu.Content>
			</DropdownMenu.Root>
		{/if}
	</div>

	<ModelsSelector
		{disabled}
		bind:this={selectorModelRef}
		currentModel={conversationModel}
		forceForegroundText={true}
		useGlobalSelection={true}
		onModelChange={handleModelChange}
	/>

	{#if isLoading}
		<Button
			type="button"
			onclick={onStop}
			class="h-8 w-8 bg-transparent p-0 hover:bg-destructive/20"
		>
			<span class="sr-only">Stop</span>
			<Square class="h-8 w-8 fill-destructive stroke-destructive" />
		</Button>
	{:else if shouldShowRecordButton}
		<ChatFormActionRecord {disabled} {hasAudioModality} {isLoading} {isRecording} {onMicClick} />
	{:else}
		<ChatFormActionSubmit
			canSend={canSend && hasModelSelected && isSelectedModelInCache}
			{disabled}
			{isLoading}
			tooltipLabel={submitTooltip}
			showErrorState={hasModelSelected && !isSelectedModelInCache}
		/>
	{/if}
</div>

<DialogMcpServersSettings
	bind:open={showMcpDialog}
	onOpenChange={(open) => (showMcpDialog = open)}
/>
