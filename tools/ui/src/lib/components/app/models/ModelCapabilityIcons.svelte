<script lang="ts">
	import { Image, Lightbulb, Mic, Video, Wrench } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
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

	let hasModalityIcons = $derived(modalities?.vision || modalities?.video || modalities?.audio);
</script>

<span class="inline-flex items-center {gapClass}">
	{#if supportsToolUse && !hideCapabilities}
		<Tooltip.Root>
			<Tooltip.Trigger>
				<Wrench class="{iconSize} text-muted-foreground" />
			</Tooltip.Trigger>

			<Tooltip.Content>
				<p>Tool use</p>
			</Tooltip.Content>
		</Tooltip.Root>
	{/if}

	{#if supportsThinking && !hideCapabilities}
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
		<span class="inline-flex items-center text-muted-foreground">
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
</span>
