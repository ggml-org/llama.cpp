<!--
	A model drop: the icon and who it is from, a big headline with the token tree beside it, the lede, use cases as
	Mono chips, and the new models as cards, the leading one tagged in cinder. Portrait is the reference; square
	drops the lede, and the story keeps the leading model only, inside Instagram's and TikTok's safe area.
-->
<script lang="ts">
	import BranchMotif from './BranchMotif.svelte';
	import ModelCard from './ModelCard.svelte';
	import { FORMATS, type ModelDropCopy } from './post';
	import PostChip from './PostChip.svelte';
	import PostFrame from './PostFrame.svelte';
	import PostHeader from './PostHeader.svelte';

	interface Props {
		copy: ModelDropCopy;
		format: 'portrait' | 'square' | 'story';
		tone: 'dark' | 'light';
	}

	let { copy, format, tone }: Props = $props();

	const size = $derived(FORMATS[format]);
	const models = $derived(format === 'story' ? copy.models.slice(0, 1) : copy.models);
</script>

<PostFrame height={size.height} label="Model drop, {size.name}, {tone}" {tone} width={size.width}>
	<div class="drop {format}">
		<PostHeader brand="icon" kicker={copy.kicker} size={34} url={copy.url} />

		<div class="top">
			<h2 class="headline">{copy.headline}</h2>

			<div class="motif"><BranchMotif /></div>
		</div>

		{#if format !== 'square'}<p class="lede">{copy.lede}</p>{/if}

		<div class="chips">
			{#each copy.chips as chip (chip)}<PostChip size={format === 'story' ? 25 : 23}
					>{chip}</PostChip
				>{/each}
		</div>

		<p class="label">{copy.section}</p>

		<div class="cards">
			{#each models as model (model.name)}<ModelCard
					{model}
					size={format === 'story' ? 44 : 38}
				/>{/each}
		</div>
	</div>
</PostFrame>

<style>
	.drop {
		display: flex;
		flex: 1;
		flex-direction: column;
		padding: 72px 76px;
	}
	.top {
		display: grid;
		grid-template-columns: 1fr 330px;
		align-items: center;
		gap: 24px;
		margin-top: 56px;
	}
	.headline {
		font-size: 88px;
	}
	.lede {
		margin-top: 36px;
		font-size: 32px;
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 12px;
		margin-top: 28px;
	}
	.label {
		margin-top: auto;
		padding-top: 40px;
		margin-bottom: 22px;
	}
	.cards {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 20px;
	}

	.square .top {
		margin-top: 48px;
		grid-template-columns: 1fr 280px;
	}
	.square .headline {
		font-size: 84px;
	}

	/* the story keeps clear of the top 220 and bottom 320 pixels, under the apps' own bars */
	.story {
		padding: 220px 84px 320px;
	}
	.story .top {
		grid-template-columns: 1fr 260px;
	}
	.story .headline {
		font-size: 100px;
	}
	.story .lede {
		font-size: 34px;
	}
	.story .cards {
		grid-template-columns: 1fr;
	}
</style>
