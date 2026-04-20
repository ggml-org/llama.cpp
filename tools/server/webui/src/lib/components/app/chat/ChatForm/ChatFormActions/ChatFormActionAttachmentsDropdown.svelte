<script lang="ts">
	import { page } from '$app/state';
	import { Plus, Settings } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { Switch } from '$lib/components/ui/switch';
	import {
		ATTACHMENT_FILE_ITEMS,
		ATTACHMENT_EXTRA_ITEMS,
		ATTACHMENT_MCP_ITEMS,
		ATTACHMENT_TOOLTIP_TEXT,
		TOOLTIP_DELAY_DURATION
	} from '$lib/constants';
	import { AttachmentMenuItemId } from '$lib/enums';
	import { ChatFormActionToolsSubmenu, McpLogo, DropdownMenuSearchable } from '$lib/components/app';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { HealthCheckStatus } from '$lib/enums';
	import type { MCPServerSettingsEntry } from '$lib/types';

	interface Props {
		class?: string;
		disabled?: boolean;
		hasAudioModality?: boolean;
		hasVisionModality?: boolean;
		hasMcpPromptsSupport?: boolean;
		hasMcpResourcesSupport?: boolean;
		onFileUpload?: () => void;
		onSystemPromptClick?: () => void;
		onMcpPromptClick?: () => void;
		onMcpSettingsClick?: () => void;
		onMcpResourcesClick?: () => void;
	}

	let {
		class: className = '',
		disabled = false,
		hasAudioModality = false,
		hasVisionModality = false,
		hasMcpPromptsSupport = false,
		hasMcpResourcesSupport = false,
		onFileUpload,
		onSystemPromptClick,
		onMcpPromptClick,
		onMcpSettingsClick,
		onMcpResourcesClick
	}: Props = $props();

	let isNewChat = $derived(!page.params.id);

	let dropdownOpen = $state(false);

	let mcpServers = $derived(mcpStore.getServersSorted().filter((s) => s.enabled));
	let hasMcpServers = $derived(mcpServers.length > 0);
	let mcpSearchQuery = $state('');
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
			mcpStore.runHealthChecksForServers(mcpServers);
		}
	}

	function handleMcpSettingsClick() {
		dropdownOpen = false;
		onMcpSettingsClick?.();
	}

	const callbacks: Record<string, (() => void) | undefined> = {
		onFileUpload: undefined,
		onSystemPromptClick: undefined,
		onMcpPromptClick: undefined,
		onMcpResourcesClick: undefined
	};

	const modalityFlags = $derived({
		hasVisionModality,
		hasAudioModality,
		hasMcpPromptsSupport,
		hasMcpResourcesSupport
	});

	function isItemEnabled(enabledWhen: string | undefined): boolean {
		if (!enabledWhen || enabledWhen === 'always') return true;
		return !!modalityFlags[enabledWhen as keyof typeof modalityFlags];
	}

	function isItemVisible(visibleWhen: string | undefined): boolean {
		if (!visibleWhen) return true;
		return !!modalityFlags[visibleWhen as keyof typeof modalityFlags];
	}

	function getSystemMessageTooltip(): string {
		return isNewChat
			? 'Add custom system message for a new conversation'
			: 'Inject custom system message at the beginning of the conversation';
	}

	$effect(() => {
		callbacks.onFileUpload = onFileUpload;
		callbacks.onSystemPromptClick = onSystemPromptClick;
		callbacks.onMcpPromptClick = () => {
			dropdownOpen = false;
			onMcpPromptClick?.();
		};
		callbacks.onMcpResourcesClick = () => {
			dropdownOpen = false;
			onMcpResourcesClick?.();
		};
	});
</script>

<div class="flex items-center gap-1 {className}">
	<DropdownMenu.Root bind:open={dropdownOpen}>
		<DropdownMenu.Trigger name="Attach files" {disabled}>
			<Tooltip.Root>
				<Tooltip.Trigger class="w-full">
					<Button
						class="file-upload-button h-8 w-8 rounded-full p-0"
						{disabled}
						variant="secondary"
						type="button"
					>
						<span class="sr-only">{ATTACHMENT_TOOLTIP_TEXT}</span>

						<Plus class="h-4 w-4" />
					</Button>
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>{ATTACHMENT_TOOLTIP_TEXT}</p>
				</Tooltip.Content>
			</Tooltip.Root>
		</DropdownMenu.Trigger>

		<DropdownMenu.Content align="start" class="w-48">
			{#each ATTACHMENT_FILE_ITEMS as item (item.id)}
				{@const enabled = isItemEnabled(item.enabledWhen)}
				{#if enabled}
					<DropdownMenu.Item
						class="{item.class ?? ''} flex cursor-pointer items-center gap-2"
						onclick={() => callbacks[item.action]?.()}
					>
						<item.icon class="h-4 w-4" />

						<span>{item.label}</span>
					</DropdownMenu.Item>
				{:else if item.disabledTooltip}
					<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
						<Tooltip.Trigger class="w-full">
							<DropdownMenu.Item
								class="{item.class ?? ''} flex cursor-pointer items-center gap-2"
								disabled
							>
								<item.icon class="h-4 w-4" />

								<span>{item.label}</span>
							</DropdownMenu.Item>
						</Tooltip.Trigger>

						<Tooltip.Content side="right">
							<p>{item.disabledTooltip}</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}
			{/each}

			{#if !isItemEnabled('hasVisionModality')}
				<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
					<Tooltip.Trigger class="w-full">
						<DropdownMenu.Item
							class="flex cursor-pointer items-center gap-2"
							onclick={() => callbacks.onFileUpload?.()}
						>
							{@const pdfItem = ATTACHMENT_FILE_ITEMS.find(
								(i) => i.id === AttachmentMenuItemId.PDF
							)}
							{#if pdfItem}
								<pdfItem.icon class="h-4 w-4" />

								<span>{pdfItem.label}</span>
							{/if}
						</DropdownMenu.Item>
					</Tooltip.Trigger>

					<Tooltip.Content side="right">
						<p>PDFs will be converted to text. Image-based PDFs may not work properly.</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/if}

			<DropdownMenu.Separator />

			{#each ATTACHMENT_EXTRA_ITEMS as item (item.id)}
				{#if item.id === AttachmentMenuItemId.SYSTEM_MESSAGE}
					<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
						<Tooltip.Trigger class="w-full">
							<DropdownMenu.Item
								class="flex cursor-pointer items-center gap-2"
								onclick={() => callbacks[item.action]?.()}
							>
								<item.icon class="h-4 w-4" />

								<span>{item.label}</span>
							</DropdownMenu.Item>
						</Tooltip.Trigger>

						<Tooltip.Content side="right">
							<p>{getSystemMessageTooltip()}</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}
			{/each}

			<ChatFormActionToolsSubmenu />

			<DropdownMenu.Sub onOpenChange={handleMcpSubMenuOpen}>
				<DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
					<McpLogo class="h-4 w-4" />

					<span>MCP Servers</span>
				</DropdownMenu.SubTrigger>

				<DropdownMenu.SubContent class="w-72 pt-0">
					<DropdownMenuSearchable
						placeholder="Search servers..."
						bind:searchValue={mcpSearchQuery}
						emptyMessage={hasMcpServers ? 'No servers found' : 'No MCP servers configured'}
						isEmpty={filteredMcpServers.length === 0}
					>
						<div class="max-h-64 overflow-y-auto">
							{#each filteredMcpServers as server (server.id)}
								{@const healthState = mcpStore.getHealthCheckState(server.id)}
								{@const hasError = healthState.status === HealthCheckStatus.ERROR}
								{@const isEnabledForChat = isServerEnabledForChat(server.id)}

								<button
									type="button"
									class="flex w-full items-center justify-between gap-2 rounded-sm px-2 py-2 text-left transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
									onclick={() => !hasError && toggleServerForChat(server.id)}
									disabled={hasError}
								>
									<div class="flex min-w-0 flex-1 items-center gap-2">
										{#if mcpStore.getServerFavicon(server.id)}
											<img
												src={mcpStore.getServerFavicon(server.id)}
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
								onclick={handleMcpSettingsClick}
							>
								<Settings class="h-4 w-4" />

								<span>Manage MCP Servers</span>
							</DropdownMenu.Item>
						{/snippet}
					</DropdownMenuSearchable>
				</DropdownMenu.SubContent>
			</DropdownMenu.Sub>

			{#each ATTACHMENT_MCP_ITEMS as item (item.id)}
				{#if isItemVisible(item.visibleWhen)}
					<DropdownMenu.Item
						class="flex cursor-pointer items-center gap-2"
						onclick={() => callbacks[item.action]?.()}
					>
						<item.icon class="h-4 w-4" />

						<span>{item.label}</span>
					</DropdownMenu.Item>
				{/if}
			{/each}
		</DropdownMenu.Content>
	</DropdownMenu.Root>
</div>
