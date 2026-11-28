<!--
	The Tools settings as the phone sheet's pages: the tool groups, one group's tools, and one tool's
	switches and the argument its calls show. The desktop draws the same as one table.
-->
<script lang="ts" module>
	import { SettingsSheetPage } from '#lib/enums/index.ts';

	export type SettingsToolsView =
		| { id: SettingsSheetPage.TOOLS }
		| { groupKey: string; id: SettingsSheetPage.TOOL_GROUP }
		| { id: SettingsSheetPage.TOOL; toolKey: string };
</script>

<script lang="ts">
	import { InsetList, InsetListRow, InsetListSplitRow } from '@ggml-org/llama-ui/app/sheets';
	import { Switch } from '@ggml-org/llama-ui/ui/switch';
	import { Check, Server } from '@lucide/svelte';
	import {
		ICON_CLASS_DEFAULT,
		TOOL_SETTINGS_TEXT,
		TOOL_SHOWN_ARG_PICK
	} from '#lib/constants/index.ts';
	import { useToolSettings } from '#lib/hooks/use-tool-settings.svelte.ts';
	import { conversationsStore, mcpStore, permissionsStore, toolsStore } from '#lib/stores/index.ts';
	import type { ToolEntry, ToolGroup } from '#lib/types/index.ts';
	import { getToolArgNames, getToolDisplay } from '#lib/utils/index.ts';

	interface Props {
		view: SettingsToolsView;
		onOpen: (view: SettingsToolsView) => void;
	}

	let { onOpen, view }: Props = $props();

	const toolSettings = useToolSettings();
	const preferences = $derived(conversationsStore.preferences);

	let groups = $derived(toolsStore.toolGroups);

	let currentGroup = $derived(
		view.id === SettingsSheetPage.TOOL_GROUP
			? groups.find((group) => group.key === view.groupKey)
			: undefined
	);
	let currentTool = $derived(
		view.id === SettingsSheetPage.TOOL
			? groups.flatMap((group) => group.tools).find((entry) => entry.key === view.toolKey)
			: undefined
	);

	function enabledCount(group: ToolGroup): number {
		return group.tools.filter((entry) => preferences.isToolEnabled(entry.key)).length;
	}
</script>

{#snippet groupSwitch(group: ToolGroup, label?: string)}
	<Switch
		aria-label={label}
		checked={preferences.isGroupChecked(group)}
		onCheckedChange={() => void preferences.toggleGroup(group)}
		size="lg"
	/>
{/snippet}

{#snippet toolSwitch(entry: ToolEntry, label?: string)}
	<Switch
		aria-label={label}
		checked={preferences.isToolEnabled(entry.key)}
		onCheckedChange={() => void preferences.toggleTool(entry.key)}
		size="lg"
	/>
{/snippet}

{#if view.id === SettingsSheetPage.TOOLS}
	{#if groups.length === 0}
		<InsetList class="px-4 py-4 text-center text-body-sm text-muted-foreground">
			{TOOL_SETTINGS_TEXT.NO_TOOLS}
		</InsetList>
	{:else}
		<InsetList caption={toolSettings.scopeNote}>
			{#each groups as group (group.key)}
				{@const favicon = group.serverId ? mcpStore.getServerFavicon(group.serverId) : null}

				<InsetListSplitRow
					onclick={() => onOpen({ groupKey: group.key, id: SettingsSheetPage.TOOL_GROUP })}
				>
					{#snippet leading()}
						{#if favicon}
							<img alt="" class="{ICON_CLASS_DEFAULT} shrink-0 rounded-sm" src={favicon} />
						{:else}
							<Server class="{ICON_CLASS_DEFAULT} shrink-0" />
						{/if}
					{/snippet}

					<span class="min-w-0 flex-1 truncate">{group.label}</span>

					<span class="text-body-sm text-muted-foreground tabular-nums">
						{enabledCount(group)}/{group.tools.length}
					</span>

					{#snippet trailing()}
						{@render groupSwitch(group, TOOL_SETTINGS_TEXT.GROUP_TOOLS(group.label))}
					{/snippet}
				</InsetListSplitRow>
			{/each}
		</InsetList>
	{/if}
{:else if view.id === SettingsSheetPage.TOOL_GROUP && currentGroup}
	{@const group = currentGroup}

	<InsetList>
		<InsetListRow as="label">
			<span class="flex-1">{TOOL_SETTINGS_TEXT.ALL_TOOLS}</span>

			{@render groupSwitch(group)}
		</InsetListRow>
	</InsetList>

	<InsetList caption={toolSettings.scopeNote}>
		{#each group.tools as entry (entry.key)}
			{@const display = getToolDisplay(entry)}

			<InsetListSplitRow
				icon={display.icon ?? undefined}
				onclick={() => onOpen({ id: SettingsSheetPage.TOOL, toolKey: entry.key })}
			>
				<span class="min-w-0 flex-1 truncate">{display.label}</span>

				{#if permissionsStore.hasTool(entry.key)}
					<span class="text-body-sm text-muted-foreground">
						{TOOL_SETTINGS_TEXT.ALWAYS_ALLOWED}
					</span>
				{/if}

				{#snippet trailing()}
					{@render toolSwitch(entry, display.label)}
				{/snippet}
			</InsetListSplitRow>
		{/each}
	</InsetList>
{:else if view.id === SettingsSheetPage.TOOL && currentTool}
	{@const entry = currentTool}
	{@const toolName = entry.definition.function.name}
	{@const pick = toolSettings.shownArgPick(toolName)}

	<InsetList>
		<InsetListRow as="label">
			<span class="flex-1">{TOOL_SETTINGS_TEXT.ENABLED}</span>

			{@render toolSwitch(entry)}
		</InsetListRow>

		<InsetListRow as="label">
			<span class="flex-1">{TOOL_SETTINGS_TEXT.ALWAYS_ALLOW}</span>

			<Switch
				checked={permissionsStore.hasTool(entry.key)}
				onCheckedChange={(checked) =>
					checked ? permissionsStore.allowTool(entry.key) : permissionsStore.revokeTool(entry.key)}
				size="lg"
			/>
		</InsetListRow>

		{#snippet caption()}
			{TOOL_SETTINGS_TEXT.ALWAYS_ALLOW_NOTE} <span class="font-mono">{toolName}</span>
		{/snippet}
	</InsetList>

	<InsetList caption={TOOL_SETTINGS_TEXT.SHOWN_ARG_NOTE} title={TOOL_SETTINGS_TEXT.SHOWN_ARG}>
		{#each [TOOL_SHOWN_ARG_PICK.DEFAULT, TOOL_SHOWN_ARG_PICK.NONE, ...getToolArgNames(entry.definition)] as option (option)}
			{@const isArgName =
				option !== TOOL_SHOWN_ARG_PICK.DEFAULT && option !== TOOL_SHOWN_ARG_PICK.NONE}

			<InsetListRow
				aria-pressed={option === pick}
				onclick={() => toolSettings.setShownArg(toolName, option)}
			>
				<span class={['min-w-0 flex-1 truncate', isArgName && 'font-mono text-body-sm']}>
					{toolSettings.shownArgLabel(option)}
				</span>

				{#if option === pick}
					<Check class="size-4 text-(--action-primary)" />
				{/if}
			</InsetListRow>
		{/each}
	</InsetList>
{/if}
