<!--
	What's new: the Llama lockup with a "new today" pill, a two-line headline with the lede beside it, a family of
	models in one card with its members inset, the other models with their figures in cinder, and the footer.
-->
<script lang="ts">
	import ModelCard from './ModelCard.svelte';
	import { FORMATS, type WhatsNewCopy } from './post';
	import PostFooter from './PostFooter.svelte';
	import PostFrame from './PostFrame.svelte';
	import PostHeader from './PostHeader.svelte';

	interface Props {
		copy: WhatsNewCopy;
		tone: 'dark' | 'light';
	}

	let { copy, tone }: Props = $props();

	const size = FORMATS.landscape;
</script>

<PostFrame height={size.height} label="What's new, {size.name}, {tone}" {tone} width={size.width}>
	<div class="news">
		<PostHeader brand="Llama" pill={copy.pill} size={32} url={copy.url} />

		<div class="top">
			<h2 class="headline">{copy.headline}</h2>

			<p class="lede">{copy.lede}</p>
		</div>

		<div class="grid">
			<ModelCard
				aside={copy.family.aside}
				model={{ about: copy.family.about, name: copy.family.name }}
				size={42}
			>
				<div class="family">
					{#each copy.family.models as model (model.name)}<ModelCard
							{model}
							size={28}
							variant="inset"
						/>{/each}
				</div>
			</ModelCard>

			<div class="side">
				{#each copy.models as model (model.name)}<ModelCard {model} size={28} />{/each}
			</div>
		</div>

		<PostFooter left={copy.footer[0]} right={copy.footer[1]} size={17} />
	</div>
</PostFrame>

<style>
	.news {
		display: flex;
		flex: 1;
		flex-direction: column;
		padding: 56px 68px 44px;
	}
	.top {
		display: grid;
		grid-template-columns: 1fr 500px;
		align-items: center;
		gap: 64px;
		margin-top: 36px;
	}
	.headline {
		font-size: 72px;
		line-height: 1.02;
		white-space: pre-line;
	}
	.lede {
		font-size: 22px;
	}
	.grid {
		display: grid;
		grid-template-columns: 1fr 500px;
		gap: 20px;
		margin-top: 36px;
		margin-bottom: 28px;
	}
	.family {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 14px;
		align-self: stretch;
		margin-top: 14px;
	}
	.side {
		display: grid;
		gap: 20px;
	}
</style>
