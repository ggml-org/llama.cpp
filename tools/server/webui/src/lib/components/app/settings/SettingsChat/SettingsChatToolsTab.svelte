<script lang="ts">
	import { ChevronDown, ChevronRight } from '@lucide/svelte';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import { TruncatedText, McpServerIdentity } from '$lib/components/app';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import { permissionsStore } from '$lib/stores/permissions.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { SvelteSet } from 'svelte/reactivity';

	let expandedGroups = new SvelteSet<string>();
	let groups = $derived(toolsStore.toolGroups);

	function toggleExpanded(label: string) {
		if (expandedGroups.has(label)) {
			expandedGroups.delete(label);
		} else {
			expandedGroups.add(label);
		}
	}
</script>

{#if groups.length === 0}
	<div class="py-8 text-center text-sm text-muted-foreground">No tools available</div>
{:else}
	<div class="space-y-2">
		{#each groups as group (group.label)}
			{@const isExpanded = expandedGroups.has(group.label)}
			<Collapsible.Root open={isExpanded} onOpenChange={() => toggleExpanded(group.label)}>
				<Collapsible.Trigger
					class="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm hover:bg-muted/50"
				>
					{#if isExpanded}
						<ChevronDown class="h-3.5 w-3.5 shrink-0" />
					{:else}
						<ChevronRight class="h-3.5 w-3.5 shrink-0" />
					{/if}

					{@const faviconUrl = group.serverId ? mcpStore.getServerFavicon(group.serverId) : null}

					<span class="inline-flex min-w-0 items-center gap-1.5 font-medium">
						<McpServerIdentity
							iconClass="h-4 w-4"
							iconRounded="rounded-sm"
							showVersion={false}
							displayName={group.label}
							{faviconUrl}
						/>
					</span>

					<span class="ml-auto shrink-0 text-xs text-muted-foreground">
						{group.tools.length} tool{group.tools.length !== 1 ? 's' : ''}
					</span>
				</Collapsible.Trigger>

				<Collapsible.Content>
					<div class="ml-4 border-l border-border/50 pl-2">
						<!-- Header row -->
						<div class="flex items-center gap-2 px-2 py-1 text-xs text-muted-foreground">
							<span class="min-w-0 flex-1">Tool</span>
							<span class="w-16 shrink-0 text-center">Enabled</span>
							<span class="w-20 shrink-0 text-center">Always allow</span>
						</div>

						{#each group.tools as tool (tool.function.name)}
							{@const toolName = tool.function.name}
							{@const isEnabled = toolsStore.isToolEnabled(toolName)}
							{@const permissionKey = toolsStore.getPermissionKey(toolName)}
							{@const isAlwaysAllowed = permissionKey
								? permissionsStore.hasTool(permissionKey)
								: false}

							<div class="flex items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-muted/50">
								<TruncatedText text={toolName} class="flex-1" showTooltip={true} />

								<div class="flex w-16 shrink-0 justify-center">
									<Checkbox
										checked={isEnabled}
										onCheckedChange={() => toolsStore.toggleTool(toolName)}
										class="h-4 w-4"
									/>
								</div>

								<div class="flex w-20 shrink-0 justify-center">
									<Checkbox
										checked={isAlwaysAllowed}
										onCheckedChange={() => {
											if (isAlwaysAllowed) {
												permissionsStore.revokeTool(permissionKey!);
											} else {
												permissionsStore.allowTool(permissionKey!);
											}
										}}
										class="h-4 w-4"
									/>
								</div>
							</div>
						{/each}
					</div>
				</Collapsible.Content>
			</Collapsible.Root>
		{/each}
	</div>
{/if}
