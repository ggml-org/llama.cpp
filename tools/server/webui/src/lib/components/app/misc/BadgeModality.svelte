<script lang="ts">
	import { ModelModality } from '$lib/enums';
	import { MODALITY_ICONS, MODALITY_LABELS } from '$lib/constants/icons';
	import { cn } from '$lib/components/ui/utils';

	type DisplayableModality = ModelModality.VISION | ModelModality.AUDIO;

	interface Props {
		modalities: ModelModality[];
		class?: string;
	}

	let { modalities, class: className = '' }: Props = $props();

	// Filter to only modalities that have icons (VISION, AUDIO)
	const displayableModalities = $derived(
		modalities.filter(
			(m): m is DisplayableModality => m === ModelModality.VISION || m === ModelModality.AUDIO
		)
	);
</script>

{#each displayableModalities as modality, index (index)}
	{@const IconComponent = MODALITY_ICONS[modality]}
	{@const label = MODALITY_LABELS[modality]}

	<span
		class={cn(
			'inline-flex items-center gap-1 rounded-md bg-muted px-2 py-1 text-xs font-medium',
			className
		)}
	>
		{#if IconComponent}
			<IconComponent class="h-3 w-3" />
		{/if}

		{label}
	</span>
{/each}
