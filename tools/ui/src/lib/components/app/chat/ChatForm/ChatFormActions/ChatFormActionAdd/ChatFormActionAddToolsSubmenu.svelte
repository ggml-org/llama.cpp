<script lang="ts">
	import { Check, ChevronDown, ChevronRight, Info, Loader2, PencilRuler } from '@lucide/svelte';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { CLI_FLAGS, ICON_CLASS_DEFAULT } from '$lib/constants';
	import { useToolsPanel } from '$lib/hooks/use-tools-panel.svelte';
	import { mcpStore, toolsStore } from '$lib/stores';
	import type { ToolGroup } from '$lib/types';

	const toolsPanel = useToolsPanel();
	const hasMcpServersAvailable = $derived(mcpStore.getServers().length > 0);
</script>

<DropdownMenu.Sub onOpenChange={(open) => open && toolsPanel.handleOpen()}>
	<DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
		<PencilRuler class={ICON_CLASS_DEFAULT} />

		<span>Tools</span>
	</DropdownMenu.SubTrigger>

	<DropdownMenu.SubContent class="w-72 p-0">
		{#if toolsPanel.totalToolCount === 0}
			{#if toolsStore.loading}
				<div class="px-3 py-4 text-center text-sm text-muted-foreground">
					<Loader2 class="mx-auto mb-1 {ICON_CLASS_DEFAULT} animate-spin" />

					Loading tools...
				</div>
			{:else if toolsStore.isToolsEndpointUnreachable}
				<div class="grid gap-2.5 px-3 py-4 text-sm text-muted-foreground">
					<span class="flex gap-2">
						<Info class="mt-0.5 {ICON_CLASS_DEFAULT} shrink-0" />

						<span>
							Run llama-server with <code>{CLI_FLAGS.TOOLS}</code> flag to enable

							<strong>Server Tools</strong>.
						</span>
					</span>

					<span class="flex gap-2">
						<Info class="mt-0.5 {ICON_CLASS_DEFAULT} shrink-0" />

						<span>
							{hasMcpServersAvailable ? 'Enable' : 'Add'} MCP Server(s) to access

							<strong>MCP Tools</strong>.
						</span>
					</span>
				</div>
			{:else if toolsStore.error}
				<div class="px-3 py-4 text-center text-sm text-muted-foreground">Failed to load tools</div>
			{:else if toolsPanel.noToolsInfoMessage}
				<div class="flex gap-2 px-3 py-4 text-sm text-muted-foreground">
					<Info class="mt-0.5 {ICON_CLASS_DEFAULT} shrink-0" />

					<span>{toolsPanel.noToolsInfoMessage}</span>
				</div>
			{:else}
				<div class="px-3 py-4 text-center text-sm text-muted-foreground">No tools available</div>
			{/if}
		{:else}
			<div class="max-h-80 overflow-y-auto p-2 pr-1">
				{#each toolsPanel.categoryGroups as group (group.key)}
					{@render groupRow(group)}
				{/each}

				{#each toolsPanel.mcpGroups as group (group.key)}
					{@render groupRow(group)}
				{/each}
			</div>
		{/if}
	</DropdownMenu.SubContent>
</DropdownMenu.Sub>

{#snippet groupRow(group: ToolGroup)}
	{@const isExpanded = toolsPanel.expandedGroups.has(group.key)}
	{@const checkState = toolsPanel.getGroupCheckState(group)}
	{@const favicon = toolsPanel.getFavicon(group)}
	{@const groupDisabled = toolsPanel.isGroupDisabled(group)}

	<Collapsible.Root
		onOpenChange={() => toolsPanel.toggleGroupExpanded(group.key)}
		open={isExpanded}
	>
		<div class="flex items-center gap-1 {groupDisabled ? 'pointer-events-none opacity-50' : ''}">
			<Collapsible.Trigger
				class="flex min-w-0 flex-1 items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-muted/50"
			>
				{#if isExpanded}
					<ChevronDown class="h-3.5 w-3.5 shrink-0" />
				{:else}
					<ChevronRight class="h-3.5 w-3.5 shrink-0" />
				{/if}

				<span class="inline-flex min-w-0 items-center gap-1.5 font-medium">
					{#if favicon}
						<img
							alt=""
							class="{ICON_CLASS_DEFAULT} shrink-0 rounded-sm"
							onerror={(e) => {
								(e.currentTarget as HTMLImageElement).style.display = 'none';
							}}
							src={favicon}
						/>
					{/if}

					<span class="truncate">{group.label}</span>
				</span>

				<span class="ml-auto shrink-0 text-xs text-muted-foreground">
					{toolsPanel.getEnabledToolCount(group)}/{group.tools.length}
				</span>
			</Collapsible.Trigger>

			<Tooltip.Root>
				<Tooltip.Trigger>
					{#snippet child({ props })}
						<Checkbox
							{...props}
							checked={checkState.checked}
							class="mr-2 {ICON_CLASS_DEFAULT} shrink-0"
							indeterminate={checkState.indeterminate}
							onCheckedChange={() => toolsPanel.toggleGroupByKey(group.key)}
						/>
					{/snippet}
				</Tooltip.Trigger>

				<Tooltip.Content side="right">
					<p>
						{checkState.checked ? 'Disable' : 'Enable'}
						{group.tools.length} tool{group.tools.length !== 1 ? 's' : ''}
					</p>
				</Tooltip.Content>
			</Tooltip.Root>
		</div>

		<Collapsible.Content>
			<div class="ml-4 flex flex-col gap-0.5 border-l border-border/50 pl-2">
				{#each group.tools as entry (entry.key)}
					{@const enabled = toolsPanel.isToolEnabled(entry)}
					{@const parentDisabled = toolsPanel.isToolParentDisabled(entry)}
					<button
						class="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm transition-colors hover:bg-muted/50 {parentDisabled
							? 'opacity-50'
							: ''}"
						onclick={() => toolsPanel.toggleTool(entry)}
						type="button"
					>
						<span
							class="flex size-4 shrink-0 items-center justify-center rounded-[4px] border border-input data-[state=checked]:border-primary data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground"
							data-slot="checkbox"
							data-state={enabled ? 'checked' : 'unchecked'}
						>
							{#if enabled}
								<Check class="size-3.5" />
							{/if}
						</span>

						<span class="min-w-0 flex-1 truncate font-mono text-[12px]">
							{entry.definition.function.name}
						</span>
					</button>
				{/each}
			</div>
		</Collapsible.Content>
	</Collapsible.Root>
{/snippet}
