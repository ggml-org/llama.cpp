<script lang="ts">
	import { Badge } from '../ui/badge';
	import * as Tooltip from '../ui/tooltip';
	import {
		CAPABILITY_ICONS,
		CAPABILITY_LABELS,
		MODALITY_FLAG_KEYS,
		MODALITY_ICONS,
		MODALITY_LABELS,
		MODALITY_ORDER
	} from '../../models/constants';
	import { ModelCapability } from '../../models/enums';
	import type { ModelModalities } from '../../models/types';

	interface Props {
		/** Name every capability in an outline badge, the way the MCP server cards do. */
		asBadges?: boolean;
		modalities?: ModelModalities;
		supportsThinking?: boolean;
		supportsToolUse?: boolean;
		hideCapabilities?: boolean;
		hideModalities?: boolean;
		iconSize?: string;
		gapClass?: string;
	}

	let {
		asBadges = false,
		gapClass = 'gap-1.25',
		hideCapabilities = false,
		hideModalities = false,
		iconSize = 'size-icon-xs',
		modalities,
		supportsThinking = false,
		supportsToolUse = false
	}: Props = $props();

	let capabilities = $derived([
		...(supportsToolUse ? [ModelCapability.TOOL_USE] : []),
		...(supportsThinking ? [ModelCapability.REASONING] : [])
	]);

	let shownModalities = $derived(
		MODALITY_ORDER.filter((modality) => modalities?.[MODALITY_FLAG_KEYS[modality]])
	);

	// an icon-less box still takes the gap where its icons would sit, so a caller with
	// nothing to show renders no box at all
	let hasIcons = $derived(
		(!hideCapabilities && capabilities.length > 0) ||
			(shownModalities.length > 0 && !hideModalities)
	);
</script>

{#if hasIcons && asBadges}
	<span class="inline-flex flex-wrap items-center gap-1.5">
		{#if !hideCapabilities}
			{#each capabilities as capability (capability)}
				{@const Icon = CAPABILITY_ICONS[capability]}

				<Badge class="h-5 gap-1 px-1.5 text-[10px]" variant="outline">
					<Icon class="size-icon-xs text-muted-foreground" />

					{CAPABILITY_LABELS[capability]}
				</Badge>
			{/each}
		{/if}

		{#if !hideModalities}
			{#each shownModalities as modality (modality)}
				{@const Icon = MODALITY_ICONS[modality]}

				<Badge class="h-5 gap-1 px-1.5 text-[10px]" variant="outline">
					<Icon class="size-icon-xs text-muted-foreground" />

					{MODALITY_LABELS[modality]}
				</Badge>
			{/each}
		{/if}
	</span>
{:else if hasIcons}
	<span class="inline-flex items-center {gapClass}">
		{#if !hideCapabilities}
			{#each capabilities as capability (capability)}
				{@const Icon = CAPABILITY_ICONS[capability]}

				<Tooltip.Root>
					<Tooltip.Trigger>
						<Icon class="{iconSize} text-muted-foreground" />
					</Tooltip.Trigger>

					<Tooltip.Content>
						<p>{CAPABILITY_LABELS[capability]}</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/each}
		{/if}

		{#if shownModalities.length > 0 && !hideModalities}
			<span class="inline-flex items-center {gapClass} text-muted-foreground">
				{#each shownModalities as modality (modality)}
					{@const Icon = MODALITY_ICONS[modality]}

					<Tooltip.Root>
						<Tooltip.Trigger>
							<Icon class={iconSize} />
						</Tooltip.Trigger>

						<Tooltip.Content>
							<p>{MODALITY_LABELS[modality]}</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/each}
			</span>
		{/if}
	</span>
{/if}
