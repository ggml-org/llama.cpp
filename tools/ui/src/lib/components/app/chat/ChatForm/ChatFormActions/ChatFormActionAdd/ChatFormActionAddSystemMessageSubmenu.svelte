<script lang="ts">
	import { MessageSquare, Plus } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { promptsStore } from '$lib/stores/prompts.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { McpLogo } from '$lib/components/app';
	import Badge from '$lib/components/ui/badge/badge.svelte';
	import { ContentPartType } from '$lib/enums';
	import { PROMPT_CONTENT_SEPARATOR } from '$lib/constants';
	import type { MCPPromptInfo, MCPServerSettingsEntry, PromptMessage } from '$lib/types';
	import { SvelteMap } from 'svelte/reactivity';

	interface Props {
		onSystemPromptClick?: () => void;
		onSystemPromptWithContent?: (content: string, instructionId?: string, title?: string) => void;
		onMcpPromptClick?: (prompt?: MCPPromptInfo) => void;
	}

	let { onSystemPromptClick, onSystemPromptWithContent, onMcpPromptClick }: Props = $props();

	let prompts = $derived(promptsStore.getPrompts());

	// Bind open state to load MCP prompts when the submenu opens
	let subOpen = $state(false);

	// MCP prompts as a flat list
	let mcpPrompts = $state<
		{
			prompt: MCPPromptInfo;
			server: MCPServerSettingsEntry | undefined;
			serverLabel: string;
			faviconUrl: string | null;
		}[]
	>([]);

	// Build server map for lookups
	let serverSettingsMap = $derived.by(() => {
		const servers = mcpStore.getServers();
		const map = new SvelteMap<string, MCPServerSettingsEntry>();
		for (const server of servers) {
			map.set(server.id, server);
		}
		return map;
	});

	async function loadMcpPrompts() {
		if (mcpPrompts.length > 0) return;

		try {
			const perChatOverrides = conversationsStore.getAllMcpServerOverrides();
			await mcpStore.ensureInitialized(perChatOverrides);

			const allPrompts = await mcpStore.getAllPrompts();

			const sortedServers = mcpStore.getServersSorted();
			const serverOrderMap = new Map(sortedServers.map((server, index) => [server.id, index]));

			const sorted = [...allPrompts].sort((a, b) => {
				const orderA = serverOrderMap.get(a.serverName) ?? Number.MAX_SAFE_INTEGER;
				const orderB = serverOrderMap.get(b.serverName) ?? Number.MAX_SAFE_INTEGER;
				return orderA - orderB;
			});

			mcpPrompts = sorted.map((prompt) => {
				const server = serverSettingsMap.get(prompt.serverName);
				return {
					prompt,
					server,
					serverLabel: server ? mcpStore.getServerLabel(server) : prompt.serverName,
					faviconUrl: server ? mcpStore.getServerFavicon(server.id) : null
				};
			});
		} catch (error) {
			console.warn('[ChatFormActionAddSystemMessageSubmenu] Failed to load MCP prompts:', error);
			mcpPrompts = [];
		}
	}

	function handleSubmenuOpen(open: boolean) {
		if (open) {
			loadMcpPrompts();
		}
	}

	function handlePromptClick(promptId: string) {
		const prompt = promptsStore.getPrompt(promptId);
		if (prompt && onSystemPromptWithContent) {
			onSystemPromptWithContent(prompt.content, promptId, prompt.title);
		}
	}

	function handleAddPromptClick() {
		onSystemPromptClick?.();
	}

	async function handleMcpPromptClick(entry: {
		prompt: MCPPromptInfo;
		server: MCPServerSettingsEntry | undefined;
		serverLabel: string;
	}) {
		// Prompts with arguments need a picker detour (for the argument form);
		// prompts without arguments post directly so the conversation gets the
		// system prompt instantly, like the library prompt branch above.
		if (entry.prompt.arguments?.length) {
			onMcpPromptClick?.(entry.prompt);
			return;
		}

		void executeAndPostAsSystemPrompt(entry.prompt);
	}

	function mcpPromptInstructionId(prompt: MCPPromptInfo): string {
		return `mcp:${prompt.serverName}:${prompt.name}`;
	}

	async function executeAndPostAsSystemPrompt(prompt: MCPPromptInfo) {
		if (!onSystemPromptWithContent) return;

		try {
			const result = await mcpStore.getPrompt(prompt.serverName, prompt.name, {});

			if (!result?.messages) return;

			const text = result.messages
				.map((msg: PromptMessage) => {
					if (typeof msg.content === 'string') return msg.content;
					if (msg.content.type === ContentPartType.TEXT) return msg.content.text;
					return '';
				})
				.filter(Boolean)
				.join(PROMPT_CONTENT_SEPARATOR);

			if (!text) return;

			onSystemPromptWithContent(text, mcpPromptInstructionId(prompt), prompt.title || prompt.name);
		} catch (error) {
			console.warn('[ChatFormActionAddSystemMessageSubmenu] Failed to execute MCP prompt:', error);
		}
	}
</script>

<DropdownMenu.Sub bind:open={subOpen} onOpenChange={handleSubmenuOpen}>
	<DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
		<MessageSquare class="h-4 w-4" />

		<span>System message</span>
	</DropdownMenu.SubTrigger>

	<DropdownMenu.SubContent class="w-72">
		<DropdownMenu.Item
			class="flex w-full cursor-pointer items-center gap-2 rounded-sm px-2 py-2 text-left transition-colors hover:bg-accent"
			onclick={handleAddPromptClick}
		>
			<Plus class="h-4 w-4" />

			<span>Add new prompt</span>
		</DropdownMenu.Item>

		{#if prompts.length > 0}
			<DropdownMenu.Separator />

			<DropdownMenu.Label class="mt-1.5 mb-1">Choose from library</DropdownMenu.Label>

			{#each prompts as prompt (prompt.id)}
				<DropdownMenu.Item
					class="flex w-full cursor-pointer flex-col items-start gap-0.5 rounded-sm px-2 py-2 text-left transition-colors hover:bg-accent"
					onclick={() => handlePromptClick(prompt.id)}
				>
					<span class="line-clamp-1 text-sm font-medium">{prompt.title}</span>
					<span class="line-clamp-3 w-full text-xs leading-5 text-muted-foreground">
						{prompt.content}
					</span>
				</DropdownMenu.Item>
			{/each}
		{/if}

		{#if mcpPrompts.length > 0}
			<DropdownMenu.Separator />

			<DropdownMenu.Label class="mt-1.5 mb-1">MCP Prompts</DropdownMenu.Label>

			{#each mcpPrompts as entry (entry)}
				<DropdownMenu.Item
					class="flex w-full cursor-pointer flex-col items-start gap-0.5 rounded-sm px-2 py-2 text-left transition-colors hover:bg-accent"
					onclick={() => handleMcpPromptClick(entry)}
				>
					<div class="flex w-full items-center gap-1.5">
						{#if entry.faviconUrl}
							<img
								src={entry.faviconUrl}
								alt=""
								class="h-3 w-3 shrink-0 rounded-sm"
								onerror={(e) => {
									(e.currentTarget as HTMLImageElement).style.display = 'none';
								}}
							/>
						{:else}
							<McpLogo class="h-3 w-3 shrink-0" />
						{/if}

						<span class="truncate text-xs text-muted-foreground">{entry.serverLabel}</span>

						{#if entry.prompt.arguments?.length}
							<Badge variant="secondary" class="ml-auto shrink-0 text-[10px]">
								{entry.prompt.arguments.length} arg{entry.prompt.arguments.length > 1 ? 's' : ''}
							</Badge>
						{/if}
					</div>

					<span class="line-clamp-1 text-sm font-medium">
						{entry.prompt.title || entry.prompt.name}
					</span>

					{#if entry.prompt.description}
						<span class="line-clamp-2 w-full text-xs leading-5 text-muted-foreground">
							{entry.prompt.description}
						</span>
					{/if}
				</DropdownMenu.Item>
			{/each}
		{/if}
	</DropdownMenu.SubContent>
</DropdownMenu.Sub>
