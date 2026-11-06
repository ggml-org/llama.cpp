<!--
	A model drop, on the Roster key visual: the icon and who it is from, the headline with the token tree beside it,
	the lede, use cases as chips, and the new models as cards, the leading one tagged in cinder. Square drops the
	lede and the chips, and the story keeps the leading model only.
-->
<script lang="ts">
	import Roster from '../layouts/Roster.svelte';
	import BranchMotif from './BranchMotif.svelte';
	import ModelCard from './ModelCard.svelte';
	import type { ModelDropCopy } from './post';

	interface Props {
		copy: ModelDropCopy;
		format: 'portrait' | 'square' | 'story';
		overlay?: boolean;
		tone: 'dark' | 'light';
	}

	let { copy, format, overlay, tone }: Props = $props();

	const models = $derived(format === 'story' ? copy.models.slice(0, 1) : copy.models);
</script>

<Roster
	chips={format === 'square' ? [] : copy.chips}
	{format}
	header={{ brand: 'icon', kicker: copy.kicker, url: copy.url }}
	headline={copy.headline}
	label={copy.section}
	lede={format === 'square' ? undefined : copy.lede}
	{overlay}
	{tone}
>
	{#snippet motif()}<BranchMotif />{/snippet}

	{#snippet cards()}
		{#each models as model (model.name)}<ModelCard {model} size={0.85} />{/each}
	{/snippet}
</Roster>
