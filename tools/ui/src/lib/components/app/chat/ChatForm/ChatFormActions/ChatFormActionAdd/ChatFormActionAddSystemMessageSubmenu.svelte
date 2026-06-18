<script lang="ts">
	import { MessageSquare, Plus } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { instructionsStore } from '$lib/stores/instructions.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { McpLogo } from '$lib/components/app';

	interface Props {
		onSystemPromptClick?: () => void;
		onSystemPromptWithContent?: (content: string, instructionId?: string, title?: string) => void;
		onMcpPromptClick?: () => void;
	}

	let { onSystemPromptClick, onSystemPromptWithContent, onMcpPromptClick }: Props = $props();

	let instructions = $derived(instructionsStore.getInstructions());

	// MCP prompts grouped by server
	let mcpPromptGroups = $state<{ serverName: string; prompts: { name: string; title: string; description: string | undefined }[] }[]>([]);
	let isLoadingMcpPrompts = $state(false);

	async function loadMcpPrompts() {
		if (mcpPromptGroups.length > 0) return;

		isLoadingMcpPrompts = true;
		try {
			const perChatOverrides = conversationsStore.getAllMcpServerOverrides();
			await mcpStore.ensureInitialized(perChatOverrides);

			const allPrompts = await mcpStore.getAllPrompts();

			const serverMap = new Map<string, { name: string; title: string; description: string | undefined }[]>();
			for (const prompt of allPrompts) {
				const list = serverMap.get(prompt.serverName) ?? [];
				list.push({
					name: prompt.name,
					title: prompt.title || prompt.name,
					description: prompt.description
				});
				serverMap.set(prompt.serverName, list);
			}

			mcpPromptGroups = [...serverMap.entries()].map(([serverName, prompts]) => ({
				serverName,
				prompts
			}));
		} catch (error) {
			console.warn('[ChatFormActionAddSystemMessageSubmenu] Failed to load MCP prompts:', error);
			mcpPromptGroups = [];
		} finally {
			isLoadingMcpPrompts = false;
		}
	}

	function handleMcpPromptsSubmenuOpen(open: boolean) {
		if (open) {
			loadMcpPrompts();
		}
	}

	function handleInstructionClick(instructionId: string) {
		const instruction = instructionsStore.getInstruction(instructionId);
		if (instruction && onSystemPromptWithContent) {
			onSystemPromptWithContent(instruction.content, instructionId, instruction.title);
		}
	}
</script>

<DropdownMenu.Root>
	<DropdownMenu.Sub onOpenChange={handleMcpPromptsSubmenuOpen}>
		<DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
			<MessageSquare class="h-4 w-4" />

			<span>System message</span>
		</DropdownMenu.SubTrigger>

		<DropdownMenu.SubContent class="w-72">
			<DropdownMenu.Item
				class="flex cursor-pointer items-center gap-2"
				onclick={() => onSystemPromptClick?.()}
			>
			    <Plus class="h-4 w-4" />

				<span>Add new instruction</span>
			</DropdownMenu.Item>

			{#if instructions.length > 0}
				<DropdownMenu.Separator />

				<DropdownMenu.Label>Choose from library</DropdownMenu.Label>

				{#each instructions as instruction (instruction.id)}
					<DropdownMenu.Item
						class="flex w-full cursor-pointer flex-col items-start gap-0.5 py-2"
						onclick={() => handleInstructionClick(instruction.id)}
					>
						<span class="line-clamp-1 text-sm font-medium">{instruction.title}</span>
						<span class="line-clamp-3 w-full text-xs leading-5 text-muted-foreground">
							{instruction.content}
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
											class="flex cursor-pointer items-center gap-2"
											onclick={() => onMcpPromptClick?.()}
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
</DropdownMenu.Root>
