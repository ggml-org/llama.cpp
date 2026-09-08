<script lang="ts">
	import { MODALITY_ICONS, MODALITY_LABELS } from '$lib/constants';
	import { ModelModality } from '$lib/enums';

	interface Props {
		modalities: ModelModality[];
		class?: string;
	}

	let { class: className = '', modalities }: Props = $props();

	const shownModalities = [ModelModality.VISION, ModelModality.AUDIO, ModelModality.VIDEO] as const;

	let visible = $derived(shownModalities.filter((modality) => modalities.includes(modality)));
</script>

{#each visible as modality (modality)}
	{@const ModalityIcon = MODALITY_ICONS[modality]}
	<span
		class={[
			'inline-flex items-center gap-1 rounded-md bg-muted px-2 py-1 text-xs font-medium',
			className
		]}
	>
		<ModalityIcon class="h-3 w-3" />

		{MODALITY_LABELS[modality]}
	</span>
{/each}
