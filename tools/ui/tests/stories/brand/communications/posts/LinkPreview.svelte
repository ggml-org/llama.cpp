<!--
	A link preview. llama.app's Open Graph card: the Llama lockup and the headline on the hero blue or the navy, the app
	in the menu bar beside them, and the caption the platforms crop least. llama.cpp's GitHub social preview: its
	lockup in Llama Mono on a neutral ground, the headline and a command in a terminal.
-->
<script lang="ts">
	import AppPanel from './AppPanel.svelte';
	import { FORMATS, type LinkPreviewCopy, type PostTone } from './post';
	import PostFrame from './PostFrame.svelte';
	import PostHeader from './PostHeader.svelte';

	interface Props {
		copy: LinkPreviewCopy;
		kind: 'og' | 'repo';
		tone: PostTone;
	}

	let { copy, kind, tone }: Props = $props();

	const size = $derived(FORMATS[kind]);
</script>

<PostFrame height={size.height} label="{size.name}, {tone}" {tone} width={size.width}>
	{#if kind === 'og'}
		<div class="og">
			<div class="copy">
				<PostHeader brand="Llama" size={40} />

				<h2 class="headline">{copy.headline}</h2>

				<p class="caption">{copy.caption}</p>
			</div>

			<div class="panel"><AppPanel /></div>
		</div>
	{:else}
		<div class="repo">
			<PostHeader brand="llama.cpp" size={44} url="github.com/ggml-org/llama.cpp" />

			<h2 class="headline">{copy.repoHeadline}</h2>

			<p class="lede">{copy.repoLede}</p>

			<pre class="terminal"><span class="prompt">$</span> {copy.command}</pre>
		</div>
	{/if}
</PostFrame>

<style>
	.og {
		display: grid;
		flex: 1;
		grid-template-columns: 1fr auto;
		align-items: start;
		gap: 40px;
		padding: 28px 48px 44px 72px;
	}
	/* the panel is drawn at its size in the menu bar; the card shows it a little closer */
	.panel {
		zoom: 1.22;
	}
	.copy {
		display: flex;
		flex-direction: column;
		height: 100%;
		padding-top: 72px;
	}
	.og .headline {
		margin-top: 36px;
		font-size: 84px;
		white-space: pre-line;
	}
	.caption {
		align-self: flex-start;
		margin: auto 0 0;
		padding: 8px 14px;
		border-radius: 8px;
		background: rgb(17 17 17 / 0.82);
		color: #fafafa;
		font-size: 24px;
		line-height: 1.2;
	}

	.repo {
		display: flex;
		flex: 1;
		flex-direction: column;
		padding: 64px 80px;
	}
	.repo .headline {
		margin-top: auto;
		font-size: 92px;
		white-space: pre-line;
	}
	.repo .lede {
		margin-top: 24px;
		font-size: 28px;
	}
	.terminal {
		margin: 40px 0 0;
		padding: 22px 26px;
		border: 2px solid var(--p-line);
		border-radius: 14px;
		background: var(--p-inset);
		font: 400 26px / 1 var(--mono);
	}
	.prompt {
		color: var(--art-cinder);
	}
</style>
