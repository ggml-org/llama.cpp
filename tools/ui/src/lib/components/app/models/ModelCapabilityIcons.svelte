<script lang="ts">
	import { Image, Lightbulb, Mic, Video, Wrench } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import type { ModelModalities } from '$lib/types/models';

	interface Props {
		/** Input modalities to render (vision / video / audio). */
		modalities?: ModelModalities;
		supportsThinking?: boolean;
		supportsToolUse?: boolean;
		/** Skip the reasoning (lightbulb) icon regardless of support. */
		hideReasoning?: boolean;
		/** Skip all modality icons regardless of support. */
		hideModalities?: boolean;
		/** Icon box size classes, e.g. `h-3 w-3` (default) or `h-4 w-4`. */
		iconSize?: string;
		/** Gap between the capability and modality icon groups. */
		gapClass?: string;
	}

	let {
		gapClass = 'gap-1.25',
		hideModalities = false,
		hideReasoning = false,
		iconSize = 'h-3 w-3',
		modalities,
		supportsThinking = false,
		supportsToolUse = false
	}: Props = $props();

	let hasModalityIcons = $derived(modalities?.vision || modalities?.video || modalities?.audio);
</script>

{#if supportsToolUse}
	<Tooltip.Root>
		<Tooltip.Trigger>
			<Wrench class="{iconSize} text-muted-foreground" />
		</Tooltip.Trigger>

		<Tooltip.Content>
			<p>Tool use</p>
		</Tooltip.Content>
	</Tooltip.Root>
{/if}

{#if supportsThinking && !hideReasoning}
	<Tooltip.Root>
		<Tooltip.Trigger>
			<Lightbulb class="{iconSize} text-muted-foreground" />
		</Tooltip.Trigger>

		<Tooltip.Content>
			<p>Reasoning</p>
		</Tooltip.Content>
	</Tooltip.Root>
{/if}

{#if hasModalityIcons && !hideModalities}
	<span class="inline-flex items-center {gapClass} text-muted-foreground">
		{#if modalities?.vision}
			<Tooltip.Root>
				<Tooltip.Trigger>
					<Image class={iconSize} />
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>Vision</p>
				</Tooltip.Content>
			</Tooltip.Root>
		{/if}

		{#if modalities?.video}
			<Tooltip.Root>
				<Tooltip.Trigger>
					<Video class={iconSize} />
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>Video</p>
				</Tooltip.Content>
			</Tooltip.Root>
		{/if}

		{#if modalities?.audio}
			<Tooltip.Root>
				<Tooltip.Trigger>
					<Mic class={iconSize} />
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>Audio</p>
				</Tooltip.Content>
			</Tooltip.Root>
		{/if}
	</span>
{/if}
