<!--
	One candidate palette: its swatches, then the same posters in its colours (numerals, tiles, a cover, the UI's own
	badges, a big word on a graph, words on a dot grid, a stacked statement, a llama.app-style popover on sage, and the
	chat: a user message over the chat form; the real components on the app's own background, as in the app). Every colour comes from
	the palette's roles, so the posters compare like for like. Set in Llama Sans Display and UI.
-->
<script lang="ts">
	import '../fonts/fonts.css';
	import { noopChatActions, userMessage } from '../../chat/messages/fixtures';
	import type { Palette } from './palettes';
	import { ChevronRight, Clock, Download, Gauge, Hash } from '@lucide/svelte';
	import { BadgeInfo, BadgesModality } from '$lib/components/app';
	import ChatMessage from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessage.svelte';
	import ChatScreenForm from '$lib/components/app/chat/ChatScreen/ChatScreenForm.svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { ModelModality } from '$lib/enums';
	import { onMount } from 'svelte';

	let { palette }: { palette: Palette } = $props();

	const r = $derived(palette.roles);
	// the chat form focuses itself when it mounts; on a still poster, hand focus back and keep the page in place
	let chat: HTMLElement | undefined = $state();

	onMount(() => {
		const id = setTimeout(() => {
			if (chat?.contains(document.activeElement)) {
				(document.activeElement as HTMLElement).blur();
				window.scrollTo(0, 0);
			}
		}, 200);

		return () => clearTimeout(id);
	});
	const dots = [
		['Run', '·', '·', '·', '·'],
		['·', 'the', '·', 'model'],
		['on', '·', 'your', '·'],
		['·', '·', 'own', '·'],
		['·', 'machine', '·'],
		['·', '·', '·', '·', '·', '·', '·']
	];
	const branches = [
		{ from: 40, nodes: [90], to: 140, y: 50 },
		{ from: 120, nodes: [175, 230], to: 290, y: 90 },
		{ from: 200, nodes: [255], to: 330, y: 130 }
	];
</script>

<div
	style:--accent={r.accent}
	style:--accent-ink={r.accentInk}
	style:--deep={r.deep}
	style:--deep-ink={r.deepInk}
	style:--deep-tone={r.deepTone}
	style:--field={r.field}
	style:--ink={r.ink}
	style:--line={r.line}
	style:--soft={r.soft}
	style:--soft-ink={r.softInk}
	style:--surface={r.surface}
	style:--warm={r.warm}
	class="palette"
>
	<ul class="swatches">
		{#each palette.swatches as s (s.name)}
			<li>
				<span style:background={s.value} class="chip"></span>{s.name}<code
					>{s.value.startsWith('#') ? s.value : 'gradient'}</code
				>
			</li>
		{/each}
	</ul>

	<div class="posters">
		<article class="card numerals">
			<p class="n">128K</p>

			<p class="n stroked">128K</p>

			<p class="n deep">128K</p>

			<p class="label">Tokens of context</p>
		</article>

		<article class="card tiles">
			<div class="tile t-accent"><span class="plus">+</span></div>

			<div class="tile t-surface"><span class="label">4-bit</span><span class="big">Q4</span></div>

			<div class="tile t-surface story"><span>Once<br />upon<br />a<br />token</span></div>

			<div class="tile t-deep"><span class="label">At home</span><span class="big">70B</span></div>
		</article>

		<article class="card cover">
			<p class="title">LOCAL</p>

			<p class="label">No. 04 · November 2026</p>

			<p class="dek">Small models,<br />big ideas.</p>

			<p class="list">Quantization, explained<br />Running 70B at home<br />The GGUF story</p>
		</article>

		<!-- the UI's own badges, on a panel of the app's own background, as they look in the app -->
		<article class="card badges">
			<div class="ui">
				<p class="tag">Badges</p>

				<div class="set">
					<Badge>Loaded</Badge>

					<Badge variant="secondary">Q4_K_M</Badge>

					<Badge variant="outline">GGUF</Badge>

					<Badge variant="destructive">Failed</Badge>
				</div>

				<div class="set">
					<BadgeInfo>
						{#snippet icon()}<Gauge class="size-icon-xs" />{/snippet}
						42.7 t/s
					</BadgeInfo>

					<BadgeInfo>
						{#snippet icon()}<Clock class="size-icon-xs" />{/snippet}
						1.2 s
					</BadgeInfo>

					<BadgeInfo>
						{#snippet icon()}<Hash class="size-icon-xs" />{/snippet}
						128K ctx
					</BadgeInfo>
				</div>

				<div class="set">
					<BadgesModality modalities={[ModelModality.VISION, ModelModality.AUDIO]} />
				</div>
			</div>
		</article>

		<article class="card gguf">
			<p class="word">GGUF</p>

			<svg aria-hidden="true" viewBox="0 0 360 150">
				<line x1="10" x2="350" y1="12" y2="12" />

				<circle cx="10" cy="12" r="7" />

				<circle cx="350" cy="12" r="7" />

				{#each branches as b (b.y)}
					<path
						d="M{b.from} 12 V{b.y - 12} Q{b.from} {b.y} {b.from + 12} {b.y} H{b.to -
							12} Q{b.to} {b.y} {b.to} {b.y - 12} V12"
					/>

					{#each b.nodes as x (x)}<circle cx={x} cy={b.y} r="7" />{/each}
				{/each}
			</svg>

			<p class="text">
				One file with a model's weights and tokenizer, quantised to fit the machine at hand.
			</p>
		</article>

		<article class="card dots">
			{#each dots as row, i (i)}
				<p class="row">
					{#each row as t, j (j)}{#if t === '·'}<i></i>{:else}<b>{t}</b>{/if}{/each}
				</p>
			{/each}
		</article>

		<article class="card statement">
			<p class="top">If it fits<br />in memory</p>

			<p class="mark"></p>

			<p class="bottom">it runs.</p>
		</article>

		<article class="card app">
			<div class="popover">
				<p class="head"><b>Llama</b><span class="dot"></span><span class="open">Open chat</span></p>

				<p class="host">localhost:9931</p>

				<p class="section">Installed models</p>

				{#each [['Qwen 2.5 7B', '4.68 GB'], ['Gemma 2 9B', '5.4 GB']] as [m, s] (m)}
					<p class="model">
						<span class="badge"></span><span class="m">{m}<small>{s}</small></span><ChevronRight
							size={14}
						/>
					</p>
				{/each}

				<p class="section">Recommended</p>

				<p class="model">
					<span class="badge accent"></span><span class="m">Llama 3.2 3B<small>2.02 GB</small></span
					><Download size={14} />
				</p>

				<p class="foot"><span>llama.cpp b9726</span><span class="btn">Settings</span></p>
			</div>
		</article>

		<!-- the chat itself: a user message with its actions over the chat form, on the app's own background -->
		<article bind:this={chat} class="card chat">
			<div class="ui">
				<p class="tag">Chat</p>

				<div class="thread">
					<ChatMessage chatActions={noopChatActions} message={userMessage} />
				</div>

				<ChatScreenForm />
			</div>
		</article>
	</div>
</div>

<style>
	.palette {
		--display: 'Llama Sans', var(--font-sans);
		--mono: ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, monospace;
		display: grid;
		gap: 14px;
	}
	.palette :is(p, ul) {
		margin: 0;
	}

	/* swatches */
	.swatches {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 18px;
		padding: 0;
		list-style: none;
		font: 500 12px / 1.25 var(--font-sans);
	}
	.swatches li {
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.swatches .chip {
		width: 22px;
		height: 22px;
		border-radius: 6px;
		box-shadow: inset 0 0 0 1px rgb(128 128 128 / 0.25);
	}
	.swatches code {
		font: 400 11px var(--mono);
		opacity: 0.6;
	}

	/* posters: cards sized by their own width, the type inside in cqi */
	.posters {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 12px;
	}
	@media (max-width: 900px) {
		.posters {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}
	.card {
		position: relative;
		display: grid;
		container-type: inline-size;
		overflow: hidden;
		box-sizing: border-box;
		aspect-ratio: 3 / 4;
		padding: 8%;
		border-radius: 10px;
		font-family: var(--display);
		box-shadow: inset 0 0 0 1px rgb(128 128 128 / 0.15);
	}
	.label {
		font: 500 3.6cqi / 1.2 var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.numerals {
		align-content: center;
		background: var(--surface);
		color: var(--ink);
	}
	.numerals .n {
		font: 700 25cqi / 0.86 var(--display);
		letter-spacing: -0.01em;
	}
	.numerals .stroked {
		color: transparent;
		-webkit-text-stroke: 0.45cqi var(--accent);
	}
	.numerals .deep {
		background: var(--deep);
		background-clip: text;
		color: transparent;
	}
	.numerals .label {
		margin-top: 4cqi;
		opacity: 0.7;
	}

	.tiles {
		grid-template-columns: 1fr 1fr;
		gap: 6%;
		align-content: center;
		background: var(--deep-tone);
	}
	.tile {
		display: grid;
		align-content: space-between;
		aspect-ratio: 1;
		padding: 7cqi;
		overflow: hidden; /* so the square holds whatever the type does */
		border-radius: 3cqi;
		container-type: inline-size;
	}
	.tile .big {
		font: 700 44cqi / 0.8 var(--display);
	}
	.tile .label {
		font-size: 8cqi;
	}
	.t-accent {
		place-items: center;
		background: var(--accent);
		color: var(--accent-ink);
	}
	.t-accent .plus {
		font: 700 80cqi / 1 var(--display);
	}
	.t-surface {
		background: var(--surface);
		color: var(--ink);
	}
	.t-surface.story {
		justify-items: end;
		align-content: end;
		text-align: right;
	}
	.t-surface.story span {
		font: 600 16cqi / 0.95 var(--display);
	}
	.t-deep {
		background: var(--deep);
		color: var(--deep-ink);
	}

	.cover {
		grid-template-rows: auto auto 1fr auto auto;
		background:
			radial-gradient(120% 80% at 85% 110%, rgb(255 255 255 / 0.1), transparent 60%), var(--deep); /* a lighter shade of its own field, no second hue */
		color: var(--deep-ink);
	}
	.cover .title {
		font: 700 27cqi / 0.86 var(--display);
		letter-spacing: -0.03em;
	}
	.cover .label {
		margin-top: 2cqi;
		opacity: 0.7;
	}
	.cover .dek {
		grid-row: 4;
		font: 700 9cqi / 1 var(--display);
	}
	.cover .list {
		margin-top: 4cqi;
		font: 400 4cqi / 1.35 var(--font-sans);
		opacity: 0.85;
	}

	/* UI on a poster looks exactly as it does in the app: the real components on a panel of the app's own background,
	   in the page's theme, so their tokens resolve as they do in the app and nothing shows through the glass */
	.ui {
		display: grid;
		gap: 14px;
		align-content: center;
		padding: 18px;
		border-radius: 14px;
		background: var(--background);
		color: var(--foreground);
		box-shadow: 0 8px 28px rgb(0 0 0 / 0.18);
		font-family: var(--font-sans);
	}
	.ui .tag {
		font: 500 11px / 1.2 var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted-foreground);
	}
	.badges {
		align-content: center;
		background: var(--deep);
	}
	.badges .set {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
	}
	.chat {
		grid-column: 1 / -1;
		aspect-ratio: auto;
		padding: 4% 8%;
		background: var(--deep);
	}
	.chat .ui {
		width: min(100%, 760px);
		margin-inline: auto;
		padding: 24px;
	}
	.chat .thread {
		display: grid;
	}

	.gguf {
		grid-template-rows: auto auto 1fr;
		gap: 5%;
		background: var(--soft);
		color: var(--soft-ink);
	}
	.gguf .word {
		font: 700 30cqi / 0.8 var(--display);
		letter-spacing: -0.04em;
	}
	.gguf svg {
		display: block;
		width: 100%;
		height: auto;
		aspect-ratio: 360 / 150;
		overflow: visible;
	}
	.gguf svg :is(line, path) {
		fill: none;
		stroke: currentColor;
		stroke-width: 2.5;
	}
	.gguf svg circle {
		fill: var(--surface);
		stroke: currentColor;
		stroke-width: 2.5;
	}
	.gguf .text {
		align-self: end;
		font: 500 4cqi / 1.4 var(--font-sans);
	}

	.dots {
		align-content: start;
		gap: 2.5%; /* of the card: its own sizes can't use cqi, which measures the card's container */
		background: var(--deep);
		color: var(--deep-ink);
	}
	.dots .row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		height: 11cqi;
	}
	.dots b {
		font: 700 11.5cqi / 1 var(--display);
	}
	.dots i {
		width: 1.6cqi;
		height: 1.6cqi;
		border-radius: 50%;
		background: color-mix(in srgb, var(--deep-ink) 30%, transparent);
	}

	.statement {
		grid-template-rows: auto 1fr auto;
		background: var(--warm);
		color: var(--ink);
	}
	.statement :is(.top, .bottom) {
		font: 700 14cqi / 0.92 var(--display);
		letter-spacing: -0.02em;
		text-transform: uppercase;
	}
	.statement .bottom {
		text-align: right;
	}
	.statement .mark {
		place-self: center;
		font: 700 30cqi / 1 var(--display);
		color: var(--accent);
	}

	/* a llama.app-style popover on the light field: the palette as UI tokens */
	.app {
		place-items: center;
		background: var(--field);
	}
	.popover {
		/* the popover is UI, so it takes the design system's neutral greys, never the palette */
		--ink: #0a0a0a;
		--line: rgb(0 0 0 / 0.12);
		--surface: #ffffff;
		--deep-tone: #737373;
		display: grid;
		gap: 2.4cqi;
		width: 100%;
		padding: 5cqi;
		border-radius: 4cqi;
		background: var(--surface);
		color: var(--ink);
		box-shadow:
			0 0 0 1px var(--line),
			0 3cqi 8cqi rgb(0 0 0 / 0.12);
		font: 400 4.2cqi / 1.25 var(--font-sans);
	}
	.popover .head {
		display: flex;
		gap: 2cqi;
		align-items: center;
	}
	.popover .dot {
		width: 1.8cqi;
		height: 1.8cqi;
		border-radius: 50%;
		background: var(--status-success); /* the UI's own online dot */
	}
	.popover .open {
		margin-left: auto;
		font-weight: 500;
		text-decoration: underline;
		text-underline-offset: 0.6cqi;
	}
	.popover .host,
	.popover small,
	.popover .section {
		color: color-mix(in srgb, var(--ink) 60%, transparent);
	}
	.popover .section {
		margin-top: 1.5cqi;
		padding-top: 2.4cqi;
		border-top: 1px solid var(--line);
		font-size: 3.6cqi;
	}
	.popover .model {
		display: flex;
		gap: 2.6cqi;
		align-items: center;
	}
	.popover .badge {
		width: 6cqi;
		height: 6cqi;
		border-radius: 50%;
		background: var(--deep-tone);
	}
	.popover .badge.accent {
		background: var(--ink);
	}
	.popover .m {
		display: grid;
		flex: 1;
		font-weight: 500;
	}
	.popover small {
		font-size: 3.4cqi;
		font-weight: 400;
	}
	.popover .foot {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-top: 1.5cqi;
		padding-top: 2.4cqi;
		border-top: 1px solid var(--line);
		font-size: 3.4cqi;
		color: color-mix(in srgb, var(--ink) 60%, transparent);
	}
	.popover .btn {
		padding: 0.8cqi 2.4cqi;
		border-radius: 1.6cqi;
		box-shadow: inset 0 0 0 1px var(--line);
		color: var(--ink);
	}
</style>
