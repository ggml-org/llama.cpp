<script lang="ts">
	import { FolderOpen, Server, Zap } from '@lucide/svelte';
	import { McpLogo } from '$lib/components/app';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { getChatFormActionsContext } from '$lib/contexts';

	const chatFormActions = getChatFormActionsContext();

	function handleServersClick() {
		chatFormActions.onMcpSettingsClick?.();
	}
</script>

<DropdownMenu.Sub>
	<DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
		<McpLogo class={ICON_CLASS_DEFAULT} />

		<span>MCP</span>
	</DropdownMenu.SubTrigger>

	<DropdownMenu.SubContent class="w-48">
		<DropdownMenu.Item class="flex cursor-pointer items-center gap-2" onclick={handleServersClick}>
			<Server class={ICON_CLASS_DEFAULT} />

			<span>Servers</span>
		</DropdownMenu.Item>

		{#if chatFormActions.hasMcpPromptsSupport}
			<DropdownMenu.Item
				class="flex cursor-pointer items-center gap-2"
				onclick={chatFormActions.onMcpPromptClick}
			>
				<Zap class={ICON_CLASS_DEFAULT} />

				<span>Prompts</span>
			</DropdownMenu.Item>
		{/if}

		{#if chatFormActions.hasMcpResourcesSupport}
			<DropdownMenu.Item
				class="flex cursor-pointer items-center gap-2"
				onclick={chatFormActions.onMcpResourcesClick}
			>
				<FolderOpen class={ICON_CLASS_DEFAULT} />

				<span>Resources</span>
			</DropdownMenu.Item>
		{/if}
	</DropdownMenu.SubContent>
</DropdownMenu.Sub>
