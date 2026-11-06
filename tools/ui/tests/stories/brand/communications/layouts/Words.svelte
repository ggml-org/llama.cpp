<!--
	Words: the copy alone, set large in Llama Mono, one phrase in it in cinder, and the sign-off at the foot: the URL,
	a line under it, and the icon. A piece that says one thing plainly, like a typed note.
-->
<script lang="ts">
	import type { Ground } from '../posts/post';
	import type { FormatKey } from '../posts/post';
	import { FORMATS, ICON } from '../posts/post';
	import PostFrame from '../posts/PostFrame.svelte';

	interface Props {
		/** the phrase in cinder, once, where it first appears */
		accent?: string;
		format: FormatKey;
		overlay?: boolean;
		paragraphs: string[];
		/** the sign-off: the URL, and a line under it */
		signoff: [string, string];
		tone: Ground;
	}

	let { accent, format, overlay, paragraphs, signoff, tone }: Props = $props();

	// each paragraph as [before, accent, after]; only the first paragraph holding the phrase gets it
	const parts = $derived.by(() => {
		let done = false;

		return paragraphs.map((p) => {
			const i = accent && !done ? p.indexOf(accent) : -1;

			if (i < 0 || !accent) return [p, '', ''];

			done = true;

			return [p.slice(0, i), accent, p.slice(i + accent.length)];
		});
	});
</script>

<PostFrame format={FORMATS[format]} label="Words, {FORMATS[format].name}, {tone}" {overlay} {tone}>
	<div class="copy">
		{#each parts as [before, mark, after], i (i)}
			<p>
				{before}{#if mark}<em>{mark}</em>{/if}{after}
			</p>
		{/each}
	</div>

	<footer>
		<p><b>{signoff[0]}</b><span>{signoff[1]}</span></p>

		<span aria-label="Llama" class="icon" role="img">{ICON}</span>
	</footer>
</PostFrame>

<style>
	.copy {
		display: grid;
		gap: calc(var(--u) * 1.4);
		margin-top: calc(var(--u) * 0.8);
	}
	.copy p {
		margin: 0;
		font: 400 calc(var(--u) * 1.6) / 1.25 var(--mono);
		letter-spacing: -0.01em;
		text-wrap: pretty;
	}
	em {
		color: var(--art-cinder);
		font-style: normal;
	}
	footer {
		display: flex;
		justify-content: space-between;
		align-items: flex-end;
		gap: var(--u);
		margin-top: auto;
	}
	footer p {
		display: grid;
		gap: calc(var(--u) * 0.2);
		margin: 0;
		font: 400 calc(var(--u) * 0.6) / 1.2 var(--mono);
	}
	footer b {
		font-weight: 500;
	}
	footer span {
		color: var(--p-muted);
	}
	.icon {
		color: var(--p-ink);
		font: 700 calc(var(--u) * 2) / 1 var(--sans);
	}
	:global(.shape-wide) .copy p {
		font-size: calc(var(--u) * 1.2);
	}
</style>
