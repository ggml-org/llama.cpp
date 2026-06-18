<script lang="ts">
	import { MessageSquare, Plus } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { promptsStore } from '$lib/stores/prompts.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { McpLogo } from '$lib/components/app';

	interface Props {
		onSystemPromptClick?: () => void;
		onSystemPromptWithContent?: (content: string, instructionId?: string, title?: string) => void;
		onMcpPromptClick?: () => void;
	}

	let { onSystemPromptClick, onSystemPromptWithContent, onMcpPromptClick }: Props = $props();

	let prompts = $derived(promptsStore.getPrompts());

	// Bind open state to load MCP prompts when the submenu opens
	let subOpen = $state(false);

	// MCP prompts grouped by server
	let mcpPromptGroups = $state<
		{
			serverName: string;
			prompts: { name: string; title: string; description: string | undefined }[];
		}[]
	>([]);

	async function loadMcpPrompts() {
		if (mcpPromptGroups.length > 0) return;

		try {
			const perChatOverrides = conversationsStore.getAllMcpServerOverrides();
			await mcpStore.ensureInitialized(perChatOverrides);

			const allPrompts = await mcpStore.getAllPrompts();

			const serverMap: Record<
				string,
				{ name: string; title: string; description: string | undefined }[]
			> = {};
			for (const prompt of allPrompts) {
				const list = serverMap[prompt.serverName] ?? [];
				list.push({
					name: prompt.name,
					title: prompt.title || prompt.name,
					description: prompt.description
				});
				serverMap[prompt.serverName] = list;
			}

			mcpPromptGroups = Object.entries(serverMap).map(([serverName, prompts]) => ({
				serverName,
				prompts
			}));
		} catch (error) {
			console.warn('[ChatFormActionAddSystemMessageSubmenu] Failed to load MCP prompts:', error);
			mcpPromptGroups = [];
		}
	}

	function handleMcpPromptsSubmenuOpen(open: boolean) {
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

	function handleMcpPromptClick() {
		onMcpPromptClick?.();
	}
</script>

<DropdownMenu.Sub bind:open={subOpen} onOpenChange={handleMcpPromptsSubmenuOpen}>
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

		{#if mcpPromptGroups.length > 0}
			<DropdownMenu.Separator />

			<DropdownMenu.Sub>
				<DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
					<McpLogo class="h-4 w-4" />

					<span>MCP Prompts</span>
				</DropdownMenu.SubTrigger>

				<DropdownMenu.SubContent class="w-72">
					{#each mcpPromptGroups as group (group.serverName)}
						<DropdownMenu.Sub>
							<DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
								<McpLogo class="h-4 w-4" />

								<span>{group.serverName}</span>
							</DropdownMenu.SubTrigger>

							<DropdownMenu.SubContent class="w-72">
								{#each group.prompts as prompt (prompt.name)}
									<DropdownMenu.Item
										class="flex w-full cursor-pointer items-center gap-2 rounded-sm px-2 py-2 text-left transition-colors hover:bg-accent"
										onclick={handleMcpPromptClick}
									>
										<span>{prompt.title}</span>
									</DropdownMenu.Item>
								{/each}
							</DropdownMenu.SubContent>
						</DropdownMenu.Sub>
					{/each}
				</DropdownMenu.SubContent>
			</DropdownMenu.Sub>
		{/if}
	</DropdownMenu.SubContent>
</DropdownMenu.Sub>
