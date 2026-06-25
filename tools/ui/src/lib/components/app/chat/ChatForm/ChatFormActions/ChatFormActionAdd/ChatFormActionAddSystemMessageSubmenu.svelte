<script lang="ts">
	import { MessageSquare, Plus } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { promptsStore } from '$lib/stores/prompts.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { McpLogo } from '$lib/components/app';
	import SearchInputMini from '$lib/components/app/forms/SearchInputMini.svelte';
	import Badge from '$lib/components/ui/badge/badge.svelte';
	import { ContentPartType } from '$lib/enums';
	import { PROMPT_CONTENT_SEPARATOR } from '$lib/constants';
	import type { MCPPromptInfo, PromptMessage } from '$lib/types';
	import { SvelteMap } from 'svelte/reactivity';

	interface Props {
		onSystemPromptClick?: () => void;
		onSystemPromptWithContent?: (content: string, instructionId?: string, title?: string) => void;
		onMcpPromptClick?: (prompt?: MCPPromptInfo) => void;
		// Kick off MCP initialization when the wrapping dropdown opens so prompts
		// are available by the time the user hovers into this submenu. The
		// result comes from `mcpStore.allPrompts`, no fetch happens here.
		preloadOnOpen?: boolean;
	}

	let {
		onSystemPromptClick,
		onSystemPromptWithContent,
		onMcpPromptClick,
		preloadOnOpen = false
	}: Props = $props();

	let prompts = $derived(promptsStore.getPrompts());

	// Single search field shared by both lists below. Matching is
	// case-insensitive across title/name, content/description, and (for MCP)
	// the server label.
	let searchQuery = $state('');

	let filteredLibraryPrompts = $derived.by(() => {
		const q = searchQuery.trim().toLowerCase();
		if (!q) return prompts;
		return prompts.filter(
			(p) => p.title.toLowerCase().includes(q) || p.content.toLowerCase().includes(q)
		);
	});

	let subOpen = $state(false);

	// Trigger MCP connection establishment as the wrapping dropdown opens.
	// Prompts are pulled eagerly inside `MCPService.connect` and surfaced
	// synchronously via `mcpStore.allPrompts`, so the submenu renders
	// without a fetch on open.
	$effect(() => {
		if (preloadOnOpen) {
			void mcpStore.ensureInitialized(conversationsStore.getAllMcpServerOverrides());
		}
	});

	let serverSettingsMap = $derived.by(() => {
		const servers = mcpStore.getServers();
		const map = new SvelteMap<string, ReturnType<typeof mcpStore.getServers>[number]>();
		for (const server of servers) {
			map.set(server.id, server);
		}
		return map;
	});

	let filteredMcpPrompts = $derived.by(() => {
		const sortedServers = mcpStore.getServersSorted();
		const serverOrderMap = new Map(sortedServers.map((server, index) => [server.id, index]));

		const entries = mcpStore.allPrompts
			.slice()
			.sort((a, b) => {
				const orderA = serverOrderMap.get(a.serverName) ?? Number.MAX_SAFE_INTEGER;
				const orderB = serverOrderMap.get(b.serverName) ?? Number.MAX_SAFE_INTEGER;
				return orderA - orderB;
			})
			.map((prompt) => {
				const server = serverSettingsMap.get(prompt.serverName);
				return {
					prompt,
					server,
					serverLabel: server ? mcpStore.getServerLabel(server) : prompt.serverName,
					faviconUrl: server ? mcpStore.getServerFavicon(server.id) : null
				};
			});

		const q = searchQuery.trim().toLowerCase();
		if (!q) return entries;
		return entries.filter((entry) => {
			const name = (entry.prompt.title || entry.prompt.name || '').toLowerCase();
			const description = (entry.prompt.description || '').toLowerCase();
			const server = entry.serverLabel.toLowerCase();
			return name.includes(q) || description.includes(q) || server.includes(q);
		});
	});

	function handlePromptClick(promptId: string) {
		const prompt = promptsStore.getPrompt(promptId);
		if (prompt && onSystemPromptWithContent) {
			onSystemPromptWithContent(prompt.content, promptId, prompt.title);
		}
	}

	function handleAddPromptClick() {
		onSystemPromptClick?.();
	}

	async function handleMcpPromptClick(entry: (typeof filteredMcpPrompts)[number]) {
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

<DropdownMenu.Sub bind:open={subOpen}>
	<DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
		<MessageSquare class="h-4 w-4" />

		<span>System message</span>
	</DropdownMenu.SubTrigger>

	<DropdownMenu.SubContent class="w-72">
		{#if prompts.length > 0 || mcpStore.allPrompts.length > 0}
			<div class="mb-1.5">
				<SearchInputMini
					bind:value={searchQuery}
					placeholder="Search by name, content, or server..."
				/>
			</div>
		{/if}

		<DropdownMenu.Item
			class="flex w-full cursor-pointer items-center gap-2 rounded-sm px-2 py-2 text-left transition-colors hover:bg-accent mt-2.5 mb-2"
			onclick={handleAddPromptClick}
		>
			<Plus class="h-4 w-4" />

			<span>Write new</span>
		</DropdownMenu.Item>

		{#if filteredLibraryPrompts.length > 0}
			<DropdownMenu.Separator />

			<DropdownMenu.Label class="mt-1.5 mb-1">Your Prompts</DropdownMenu.Label>

			{#each filteredLibraryPrompts as prompt (prompt.id)}
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

		{#if filteredMcpPrompts.length > 0}
			<DropdownMenu.Separator />

			<DropdownMenu.Label class="mt-1.5 mb-1">MCP Prompts</DropdownMenu.Label>

			{#each filteredMcpPrompts as entry (entry)}
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

		{#if searchQuery.trim() !== '' && (prompts.length > 0 || mcpStore.allPrompts.length > 0) && filteredLibraryPrompts.length === 0 && filteredMcpPrompts.length === 0}
			<div class="px-2 py-2 text-xs text-muted-foreground">No matches</div>
		{/if}
	</DropdownMenu.SubContent>
</DropdownMenu.Sub>
