<script lang="ts">
	import { PencilRuler, ChevronDown, ChevronRight, Loader2 } from '@lucide/svelte';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import { useToolsPanel } from '$lib/hooks/use-tools-panel.svelte';

	const {
		expandedGroups,
		groups,
		activeGroups,
		totalToolCount,
		getGroupCheckedState,
		getEnabledToolCount,
		getFavicon,
		toggleGroupExpanded,
		handleOpen
	} = useToolsPanel();
</script>

<DropdownMenu.Sub onOpenChange={(open) => open && handleOpen()}>
	<DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
		<PencilRuler class="h-4 w-4" />

		<span>Tools</span>
	</DropdownMenu.SubTrigger>

	<DropdownMenu.SubContent class="w-72 p-0">
		{#if totalToolCount === 0 && groups.length === 0}
			<div class="px-3 py-4 text-center text-sm text-muted-foreground">
				{#if toolsStore.loading}
					<Loader2 class="mx-auto mb-1 h-4 w-4 animate-spin" />
					Loading tools...
				{:else if toolsStore.error}
					Failed to load tools
				{:else}
					No tools available
				{/if}
			</div>
		{:else}
			<div class="max-h-80 overflow-y-auto p-2 pr-1">
				{#each activeGroups as group (group.label)}
					{@const isExpanded = expandedGroups.has(group.label)}
					{@const { checked, indeterminate } = getGroupCheckedState(group)}
					{@const favicon = getFavicon(group)}

					<Collapsible.Root open={isExpanded} onOpenChange={() => toggleGroupExpanded(group.label)}>
						<div class="flex items-center gap-1">
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
											src={favicon}
											alt=""
											class="h-4 w-4 shrink-0 rounded-sm"
											onerror={(e) => {
												(e.currentTarget as HTMLImageElement).style.display = 'none';
											}}
										/>
									{/if}

									<span class="truncate">{group.label}</span>
								</span>

								<span class="ml-auto shrink-0 text-xs text-muted-foreground">
									{getEnabledToolCount(group)}/{group.tools.length}
								</span>
							</Collapsible.Trigger>

							<Tooltip.Root>
								<Tooltip.Trigger>
									<Checkbox
										{checked}
										{indeterminate}
										onCheckedChange={() => toolsStore.toggleGroup(group)}
										class="mr-2 h-4 w-4 shrink-0"
									/>
								</Tooltip.Trigger>

								<Tooltip.Content side="right">
									<p>
										{checked ? 'Disable' : 'Enable'}
										{group.tools.length} tool{group.tools.length !== 1 ? 's' : ''}
									</p>
								</Tooltip.Content>
							</Tooltip.Root>
						</div>

						<Collapsible.Content>
							<div class="ml-4 flex flex-col gap-0.5 border-l border-border/50 pl-2">
								{#each group.tools as tool (tool.function.name)}
									<button
										type="button"
										class="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm transition-colors hover:bg-muted/50"
										onclick={() => toolsStore.toggleTool(tool.function.name)}
									>
										<Checkbox
											checked={toolsStore.isToolEnabled(tool.function.name)}
											onCheckedChange={() => toolsStore.toggleTool(tool.function.name)}
											class="h-4 w-4 shrink-0"
										/>

										<span class="min-w-0 flex-1 truncate font-mono text-[12px]">
											{tool.function.name}
										</span>
									</button>
								{/each}
							</div>
						</Collapsible.Content>
					</Collapsible.Root>
				{/each}
			</div>
		{/if}
	</DropdownMenu.SubContent>
</DropdownMenu.Sub>
