<script lang="ts">
	import { Checkbox } from '@ggml-org/llama-ui/ui/checkbox';
	import * as Collapsible from '@ggml-org/llama-ui/ui/collapsible';
	import * as Select from '@ggml-org/llama-ui/ui/select';
	import { ChevronDown, ChevronRight } from '@lucide/svelte';
	import { McpServerIdentity, TruncatedText } from '#lib/components/app/index.ts';
	import {
		ICON_CLASS_DEFAULT,
		SETTINGS_KEYS,
		TOOL_CALL_SHOWN_ARG_NONE
	} from '#lib/constants/index.ts';
	import { ToolSource } from '#lib/enums/tools.enums.ts';
	import {
		conversationsStore,
		mcpStore,
		permissionsStore,
		settingsStore,
		toolsStore
	} from '#lib/stores/index.ts';
	import type { OpenAIToolDefinition } from '#lib/types/index.ts';
	import { getToolUi, parseToolCallShownArgOverrides } from '#lib/utils/index.ts';
	import { SvelteSet } from 'svelte/reactivity';

	// the shown-argument picks: a tool follows the global setting until it gets one of its own
	const SHOWN_ARG_DEFAULT = '__default__';
	const SHOWN_ARG_NONE = '__none__';

	let expandedGroups = new SvelteSet<string>();
	let groups = $derived(toolsStore.toolGroups);
	let shownArgOverrides = $derived(
		parseToolCallShownArgOverrides(
			settingsStore.config[SETTINGS_KEYS.TOOL_CALL_SHOWN_ARG_OVERRIDES]
		)
	);
	let globalShownArg = $derived(
		String(settingsStore.config[SETTINGS_KEYS.TOOL_CALL_SHOWN_ARG] ?? '').trim()
	);

	function argNames(definition: OpenAIToolDefinition): string[] {
		const properties = (definition.function.parameters as { properties?: object } | undefined)
			?.properties;

		return properties ? Object.keys(properties) : [];
	}

	function shownArgPick(toolName: string): string {
		const own = shownArgOverrides[toolName];

		if (own === undefined) return SHOWN_ARG_DEFAULT;

		return own === TOOL_CALL_SHOWN_ARG_NONE ? SHOWN_ARG_NONE : own;
	}

	function shownArgLabel(pick: string): string {
		if (pick === SHOWN_ARG_NONE) return 'None';

		if (pick === SHOWN_ARG_DEFAULT)
			return globalShownArg ? `Default (${globalShownArg})` : 'Default';

		return pick;
	}

	function setShownArg(toolName: string, pick: string) {
		const next = { ...shownArgOverrides };

		if (pick === SHOWN_ARG_DEFAULT) delete next[toolName];
		else next[toolName] = pick === SHOWN_ARG_NONE ? TOOL_CALL_SHOWN_ARG_NONE : pick;

		settingsStore.updateConfig(SETTINGS_KEYS.TOOL_CALL_SHOWN_ARG_OVERRIDES, JSON.stringify(next));
	}

	function toggleExpanded(key: string) {
		if (expandedGroups.has(key)) {
			expandedGroups.delete(key);
		} else {
			expandedGroups.add(key);
		}
	}
</script>

{#if groups.length === 0}
	<div class="py-8 text-center text-body-sm text-muted-foreground">No tools available</div>
{:else}
	<div class="space-y-2">
		<p class="text-body-sm text-muted-foreground">
			{#if conversationsStore.activeConversation}
				Applies to this conversation.
			{:else}
				Applies to new conversations. Tool picks inside a chat only affect that chat.
			{/if}
		</p>

		{#each groups as group (group.key)}
			{@const isExpanded = expandedGroups.has(group.key)}
			<Collapsible.Root onOpenChange={() => toggleExpanded(group.key)} open={isExpanded}>
				<Collapsible.Trigger
					class="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-body-sm hover:bg-muted/50"
				>
					{#if isExpanded}
						<ChevronDown class="size-icon-sm shrink-0" />
					{:else}
						<ChevronRight class="size-icon-sm shrink-0" />
					{/if}

					<Checkbox
						checked={conversationsStore.preferences.isGroupChecked(group)}
						onCheckedChange={() => void conversationsStore.preferences.toggleGroup(group)}
						onclick={(e) => e.stopPropagation()}
					/>

					{@const faviconUrl = group.serverId ? mcpStore.getServerFavicon(group.serverId) : null}

					<span class="inline-flex min-w-0 items-center gap-1.5 font-medium">
						{#if group.source === 'mcp'}
							<McpServerIdentity
								displayName={group.label}
								{faviconUrl}
								iconClass={ICON_CLASS_DEFAULT}
								iconRounded="rounded-sm"
								showVersion={false}
							/>
						{:else}
							<TruncatedText class="font-medium" text={group.label} />
						{/if}
					</span>

					<span class="ml-auto shrink-0 text-caption text-muted-foreground">
						{group.tools.length} tool{group.tools.length !== 1 ? 's' : ''}
					</span>
				</Collapsible.Trigger>

				<Collapsible.Content>
					<div class="ml-4 border-l border-border/50 pl-2">
						<!-- Header row -->
						<div class="flex items-center gap-2 px-2 py-1 text-caption text-muted-foreground">
							<span class="min-w-0 flex-1">Tool</span>

							<span class="w-16 shrink-0 text-center">Enabled</span>

							<span class="w-20 shrink-0 text-center">Always allow</span>

							<span
								class="w-32 shrink-0"
								title="The argument a call's line shows when the call has it"
							>
								Shown argument
							</span>
						</div>

						{#each group.tools as entry (entry.key)}
							{@const toolName = entry.definition.function.name}
							{@const toolUi =
								entry.source === ToolSource.SERVER || entry.source === ToolSource.BROWSER
									? getToolUi(toolName)
									: null}
							{@const displayLabel = toolUi?.label ?? toolName}
							{@const IconComponent = toolUi?.icon ?? null}
							{@const isEnabled = conversationsStore.preferences.isToolEnabled(entry.key)}
							{@const permissionKey = entry.key}
							{@const isAlwaysAllowed = permissionsStore.hasTool(permissionKey)}
							{@const shownArg = shownArgPick(toolName)}

							<div
								class="flex items-center gap-2 rounded px-2 py-1.5 text-body-sm hover:bg-muted/50"
							>
								<span class="flex min-w-0 flex-1 items-center gap-1.5">
									{#if IconComponent}
										<IconComponent class={ICON_CLASS_DEFAULT} />
									{/if}

									<TruncatedText class="min-w-0" showTooltip={true} text={displayLabel} />
								</span>

								<div class="flex w-16 shrink-0 justify-center">
									<Checkbox
										checked={isEnabled}
										class={ICON_CLASS_DEFAULT}
										onCheckedChange={() =>
											void conversationsStore.preferences.toggleTool(entry.key)}
									/>
								</div>

								<div class="flex w-20 shrink-0 justify-center">
									<Checkbox
										checked={isAlwaysAllowed}
										class={ICON_CLASS_DEFAULT}
										onCheckedChange={() => {
											if (isAlwaysAllowed) {
												permissionsStore.revokeTool(permissionKey);
											} else {
												permissionsStore.allowTool(permissionKey);
											}
										}}
									/>
								</div>

								<div class="w-32 shrink-0">
									<Select.Root
										onValueChange={(value) => setShownArg(toolName, value)}
										type="single"
										value={shownArg}
									>
										<Select.Trigger
											aria-label="Argument shown for {displayLabel}"
											class="h-7 w-full px-2 text-caption"
											size="sm"
										>
											<span class="truncate">
												{shownArg === SHOWN_ARG_DEFAULT ? 'Default' : shownArgLabel(shownArg)}
											</span>
										</Select.Trigger>

										<Select.Content>
											<Select.Item
												label={shownArgLabel(SHOWN_ARG_DEFAULT)}
												value={SHOWN_ARG_DEFAULT}
											>
												{shownArgLabel(SHOWN_ARG_DEFAULT)}
											</Select.Item>

											<Select.Item label="None" value={SHOWN_ARG_NONE}>None</Select.Item>

											{#each argNames(entry.definition) as arg (arg)}
												<Select.Item label={arg} value={arg}>
													<span class="font-mono">{arg}</span>
												</Select.Item>
											{/each}
										</Select.Content>
									</Select.Root>
								</div>
							</div>
						{/each}
					</div>
				</Collapsible.Content>
			</Collapsible.Root>
		{/each}
	</div>
{/if}
