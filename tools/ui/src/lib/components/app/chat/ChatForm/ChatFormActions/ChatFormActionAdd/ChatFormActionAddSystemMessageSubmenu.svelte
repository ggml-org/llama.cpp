<script lang="ts">
	import { MessageSquare, Plus } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { skillsStore } from '$lib/stores/skills.svelte';
	import { skillPreferencesStore } from '$lib/stores/skill-preferences.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { McpLogo } from '$lib/components/app';
	import SearchInput from '$lib/components/app/forms/SearchInput.svelte';
	import Badge from '$lib/components/ui/badge/badge.svelte';
	import { ContentPartType } from '$lib/enums';
	import { PROMPT_CONTENT_SEPARATOR } from '$lib/constants';
	import { buildMcpPromptId } from '$lib/utils';
	import type { MCPPromptInfo, PromptMessage } from '$lib/types';
	import { SvelteMap } from 'svelte/reactivity';

	interface Props {
		onSystemPromptClick?: () => void;
		onSystemPromptWithContent?: (content: string, skillId?: string, title?: string) => void;
		onMcpPromptClick?: (prompt?: MCPPromptInfo) => void;
		preloadOnOpen?: boolean;
	}

	let {
		onSystemPromptClick,
		onSystemPromptWithContent,
		onMcpPromptClick,
		preloadOnOpen = false
	}: Props = $props();

	let skills = $derived(skillsStore.getSkills());

	let searchQuery = $state('');

	let filteredLibrarySkills = $derived.by(() => {
		const q = searchQuery.trim().toLowerCase();
		if (!q) return skills;
		return skills.filter(
			(s) =>
				s.name?.toLowerCase().includes(q) ||
				s.content?.toLowerCase().includes(q) ||
				(s.description?.toLowerCase().includes(q) ?? false)
		);
	});

	let subOpen = $state(false);

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

	function handleSkillClick(skillId: string) {
		const skill = skillsStore.getSkill(skillId);
		if (skill && onSystemPromptWithContent) {
			onSystemPromptWithContent(skill.content ?? '', skillId, skill.name ?? skillId);
		}
	}

	function handleAddSkillClick() {
		onSystemPromptClick?.();
	}

	async function handleMcpPromptClick(entry: (typeof filteredMcpPrompts)[number]) {
		if (entry.prompt.arguments?.length) {
			onMcpPromptClick?.(entry.prompt);
			return;
		}

		void executeAndPostAsSystemPrompt(entry.prompt);
	}

	function mcpPromptId(prompt: MCPPromptInfo): string {
		return buildMcpPromptId(prompt.serverName, prompt.name);
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

			onSystemPromptWithContent(text, mcpPromptId(prompt), prompt.title || prompt.name);
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
		{#if skills.length > 0 || mcpStore.allPrompts.length > 0}
			<div class="mb-1.5">
				<SearchInput
					bind:value={searchQuery}
					variant="sm"
					placeholder="Search by name, content, or server..."
				/>
			</div>
		{/if}

		<DropdownMenu.Item
			class="flex w-full cursor-pointer items-center gap-2 rounded-sm px-2 py-2 text-left transition-colors hover:bg-accent mt-2.5 mb-2"
			onclick={handleAddSkillClick}
		>
			<Plus class="h-4 w-4" />

			<span>Write new</span>
		</DropdownMenu.Item>

		{#if filteredLibrarySkills.length > 0}
			<DropdownMenu.Separator />

			<DropdownMenu.Label class="mt-1.5 mb-1">Your Skills</DropdownMenu.Label>

			{#each filteredLibrarySkills as skill (skill.id)}
				<DropdownMenu.Item
					class="flex w-full cursor-pointer flex-col items-start gap-0.5 rounded-sm px-2 py-2 text-left transition-colors hover:bg-accent"
					onclick={() => handleSkillClick(skill.id)}
				>
					<span class="line-clamp-1 text-sm font-medium">{skill.name ?? skill.id}</span>
					{#if skillPreferencesStore.isAlwaysOn(skill.id)}
						<span class="ml-1 text-[10px] uppercase tracking-wide text-muted-foreground"
							>always-on</span
						>
					{/if}
					{#if skill.description}
						<span class="line-clamp-1 w-full text-xs leading-5 text-muted-foreground">
							{skill.description}
						</span>
					{/if}
					<span class="line-clamp-3 w-full text-xs leading-5 text-muted-foreground">
						{skill.content}
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

		{#if searchQuery.trim() !== '' && (skills.length > 0 || mcpStore.allPrompts.length > 0) && filteredLibrarySkills.length === 0 && filteredMcpPrompts.length === 0}
			<div class="px-2 py-2 text-xs text-muted-foreground">No matches</div>
		{/if}
	</DropdownMenu.SubContent>
</DropdownMenu.Sub>
