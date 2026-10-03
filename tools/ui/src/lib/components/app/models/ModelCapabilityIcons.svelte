<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import {
		CAPABILITY_ICONS,
		CAPABILITY_LABELS,
		MODALITY_FLAG_KEYS,
		MODALITY_ICONS,
		MODALITY_LABELS,
		MODALITY_ORDER
	} from '$lib/constants';
	import { ModelCapability } from '$lib/enums';
	import type { ModelModalities } from '$lib/types/models';

	interface Props {
		modalities?: ModelModalities;
		supportsThinking?: boolean;
		supportsToolUse?: boolean;
		hideCapabilities?: boolean;
		hideModalities?: boolean;
		iconSize?: string;
		gapClass?: string;
	}

	let {
		gapClass = 'gap-1.25',
		hideCapabilities = false,
		hideModalities = false,
		iconSize = 'h-3 w-3',
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
</script>

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
