<!--
	Llama for Windows' tray flyout, measured off a screenshot of the app (Morgan's build, November 2026; Windows 11
	dark at 200%, so every size here is half the screenshot's pixels): the icon with settings and quit; the Models card,
	a fixed height that clips its list, with the installed models (a chip, the name, tags for size, quant, vision and
	file size, and folder, delete and run) and, under "Browse more", llama.app's catalog with each maker's logo (the
	macOS app's ModelLogos); the Hugging Face search; and the server's status with its build. The catalog's lines come
	without the app's stray Markdown asterisks. `card` draws the Models card alone, for a close-up. `view="details"`
	shows a model's page in the card instead (after the 0.13 release notes' screenshot): its sizes, and its versions with
	the fit verdict for this PC, fits or might fit.
-->
<script lang="ts">
	import hfLogo from '../../../fixtures/assets/hf-logo.svg?url';
	import { PATHS, SIZE } from '../../logo-showcase/lib/logo';
	import gemma from './logos/gemma.svg?raw';
	import gpt from './logos/gpt.svg?raw';
	import liquid from './logos/liquid.svg?raw';
	import meta from './logos/meta.svg?raw';
	import qwen from './logos/qwen.svg?raw';
	import {
		ArrowLeft,
		Check,
		ChevronRight,
		Cpu,
		Download,
		Eye,
		Folder,
		Monitor,
		Play,
		Power,
		Search,
		Settings,
		SquareArrowOutUpRight,
		Trash2
	} from '@lucide/svelte';

	interface Props {
		/** the Models card alone */
		card?: boolean;
		/** the model list, or a model's page with its versions and their fit verdicts */
		view?: 'details' | 'list';
	}

	let { card = false, view = 'list' }: Props = $props();

	// LFM 2.5's page as the app shows it: one size, two versions, each with whether it fits this PC
	const versions = [
		{ fit: 'maybe', label: 'Higher quality', quant: 'Q8_0', size: '25.5 GB' },
		{ fit: 'yes', label: 'Balanced', quant: 'Q4_K_M', size: '14.5 GB' }
	] as const;

	// small and medium models from llama.app's catalog
	const installed = [
		{ name: 'Gemma 4', size: '4.59 GB', tags: ['E4B', 'Q4_K_M'] },
		{ name: 'LFM 3 Omni', size: '1.8 GB', tags: ['2B', 'Q8_0'] }
	];
	// the logos as the app shows them: most in white on the tile, Meta's and Liquid's on a white plate of their own
	const catalog = [
		{
			about: 'OpenAI’s open-weight family for local reasoning and agents.',
			logo: gpt,
			name: 'GPT-OSS',
			sizes: '20B · 120B'
		},
		{
			about: 'Google’s most capable open agentic models for your laptops.',
			logo: gemma,
			name: 'Gemma 4',
			sizes: 'E2B · E4B · 12B · 26B-A4B · 31B'
		},
		{
			about: 'Alibaba’s multimodal mid-sized models for coding and research.',
			logo: qwen,
			name: 'Qwen 3.8',
			sizes: '27B'
		},
		{
			about: 'Meta’s local model to run on your laptop powering always-on agents.',
			logo: meta,
			name: 'Muse Glimmer',
			plate: 'square',
			sizes: '30B'
		},
		{
			about: 'Liquid AI’s fast local model for reasoning and tool use.',
			logo: liquid,
			name: 'LFM 2.5',
			plate: 'round',
			sizes: '24B-A2B'
		},
		{
			about: 'Liquid AI’s tiny omni model for text, images, and audio.',
			logo: liquid,
			name: 'LFM 3 Omni',
			plate: 'round',
			sizes: '2B'
		}
	];
</script>

{#snippet models()}
	<div class="card">
		<p class="card-title">
			Models <span class="open"><SquareArrowOutUpRight size={13} strokeWidth={1.5} /></span>
		</p>

		{#each installed as m (m.name)}
			<div class="row installed">
				<span class="tile"><Cpu size={18} strokeWidth={1.75} /></span>

				<span class="info">
					<b>{m.name}</b>

					<span class="tags">
						{#each m.tags as tag (tag)}<i>{tag}</i>{/each}

						<i aria-label="vision"><Eye size={11} strokeWidth={1.5} /></i>

						<span>{m.size}</span>
					</span>
				</span>

				<span class="actions">
					<Folder size={17} strokeWidth={1.2} />

					<Trash2 size={17} strokeWidth={1.2} />

					<Play size={17} strokeWidth={1.2} />
				</span>
			</div>
		{/each}

		<p class="browse">Browse more</p>

		{#each catalog as m (m.name)}
			<div class="row">
				<span class="tile">
					<span class={['logo', m.plate]}>{@html m.logo}</span>
				</span>

				<span class="info">
					<b>{m.name}<span class="bar">|</span><span class="sizes">{m.sizes}</span></b>

					<small>{m.about}</small>
				</span>

				<ChevronRight size={13} strokeWidth={1.5} />
			</div>
		{/each}
	</div>
{/snippet}

{#snippet details()}
	<div class="card details">
		<p class="back"><ArrowLeft size={15} strokeWidth={1.5} />Models</p>

		<div class="model">
			<span class="tile big"><span class="logo round">{@html liquid}</span></span>

			<span class="info">
				<b>LFM 2.5</b>

				<small class="clamp"
					>Liquid AI’s reasoning mixture-of-experts model with 24B parameters and 2B active, built
					for fast on-device agents.</small
				>
			</span>
		</div>

		<p class="label">Choose a model size</p>

		<span class="size">24B-A2B</span>

		<p class="label">Choose a version</p>

		{#each versions as v (v.quant)}
			<div class="version">
				<Download size={16} strokeWidth={1.25} />

				<span class="name">{v.label}</span>

				<span class="weight">{v.quant} · {v.size}</span>

				<span
					aria-label={v.fit === 'yes' ? 'Fits this PC' : 'Might fit this PC'}
					class={['fit', v.fit]}
					role="img"
				>
					{#if v.fit === 'yes'}<Check size={9} strokeWidth={3} />{:else}?{/if}
				</span>
			</div>
		{/each}
	</div>
{/snippet}

{#snippet body()}
	{#if view === 'details'}{@render details()}{:else}{@render models()}{/if}
{/snippet}

{#if card}
	<div class="flyout only-card">{@render body()}</div>
{:else}
	<div class="flyout">
		<div class="head">
			<svg aria-label="Llama" class="icon" role="img" viewBox="0 0 {SIZE} {SIZE}">
				{#each PATHS as d (d)}<path {d} />{/each}
			</svg>

			<span class="buttons"
				><Settings size={15} strokeWidth={1.5} /><Power size={15} strokeWidth={1.5} /></span
			>
		</div>

		{@render body()}

		<div class="search">
			<span class="hf"><img alt="Hugging Face" src={hfLogo} /></span>

			<span class="field">Search models …<Search size={12} strokeWidth={1.75} /></span>
		</div>

		<p class="status">
			<i class="dot"></i>0.14.1.0 - version: 0.8.0-dev (build 12019, commit 1052f7e1d)<Monitor
				size={13}
				strokeWidth={1.75}
			/>
		</p>
	</div>
{/if}

<style>
	/* a model's page in the card: measured off the 0.13 release notes' screenshot, at half its pixels */
	.details {
		padding: 10px 16px 16px 24px;
	}
	.back {
		display: flex;
		align-items: center;
		gap: 8px;
		height: 31.5px;
		margin: 0;
		color: rgb(255 255 255 / 0.82);
		font-size: 13.5px;
	}
	.model {
		display: flex;
		align-items: flex-start;
		gap: 10px;
		margin: 10px 0 0 -1.5px;
	}
	.tile.big {
		width: 40px;
		height: 40px;
	}
	.tile.big .logo.round {
		width: 28px;
		height: 28px;
		padding: 5px;
	}
	.clamp {
		display: -webkit-box;
		-webkit-box-orient: vertical;
		-webkit-line-clamp: 2;
		line-clamp: 2;
		white-space: normal !important;
	}
	.label {
		margin: 16px 0 8px;
		color: rgb(255 255 255 / 0.85);
		font-size: 13px;
		font-weight: 600;
	}
	.size {
		display: inline-grid;
		place-items: center;
		align-self: flex-start;
		height: 29px;
		padding: 0 8px;
		box-sizing: border-box;
		border: 1.5px solid #4cc2ff;
		border-radius: 4px;
		background: #3c3d3f;
		font-size: 13px;
		font-weight: 600;
	}
	.version {
		display: flex;
		align-items: center;
		gap: 12px;
		height: 30px;
		padding: 0 4px;
		color: rgb(255 255 255 / 0.85);
	}
	.version .name {
		flex: 1;
		color: #fff;
		font-size: 14px;
	}
	.weight {
		font-size: 13px;
	}
	.fit {
		display: grid;
		place-items: center;
		width: 12.5px;
		height: 12.5px;
		border-radius: 2.5px;
		font:
			700 10px / 1 'Segoe UI Variable Text',
			'Segoe UI',
			system-ui,
			sans-serif;
	}
	.fit.maybe {
		background: #fe9a00;
		color: #1a1a1a;
	}
	.fit.yes {
		background: #009966;
		color: #fff;
	}

	/* Windows 11's dark acrylic flyout, its card a step lighter, in Segoe UI where there is one */
	.flyout {
		display: flex;
		flex-direction: column;
		width: 417px;
		height: 560px;
		box-sizing: border-box;
		border: 1px solid rgb(255 255 255 / 0.16);
		border-radius: 8px;
		background: #242422;
		color: #fff;
		font:
			400 12.5px / 1.3 'Segoe UI Variable Text',
			'Segoe UI',
			system-ui,
			sans-serif;
		box-shadow: 0 16px 40px rgb(0 0 0 / 0.4);
	}
	.only-card {
		width: auto;
		height: auto;
		border: 0;
		background: none;
		box-shadow: none;
	}
	.head {
		display: flex;
		flex: none;
		align-items: center;
		justify-content: space-between;
		height: 44px;
		margin: 0 15px;
		border-bottom: 1px solid rgb(255 255 255 / 0.12);
	}
	.icon {
		width: 20px;
		height: 20px;
		fill: #fff;
	}
	.buttons {
		display: flex;
		gap: 13px;
	}

	.card {
		display: flex;
		flex: none;
		flex-direction: column;
		width: 386px;
		height: 410px;
		box-sizing: border-box;
		margin: 13px 15px 0 16px;
		overflow: hidden;
		border-radius: 8px;
		background: #31302d;
		box-shadow:
			0 0 0 1px rgb(0 0 0 / 0.18),
			0 1px 2px rgb(0 0 0 / 0.3);
	}
	.card-title {
		display: flex;
		flex: none;
		align-items: center;
		gap: 10px;
		height: 31.5px;
		margin: 0;
		padding: 7px 0 0 9px;
		box-sizing: border-box;
		color: rgb(255 255 255 / 0.82);
		font-size: 13.5px;
	}
	.open {
		display: grid;
		color: rgb(255 255 255 / 0.75);
	}

	.row {
		position: relative;
		display: flex;
		flex: none;
		align-items: center;
		gap: 10.5px;
		height: 45.5px;
		padding: 0 19px 0 16px;
		color: #fff;
	}
	.installed {
		height: 48px;
		padding-right: 22px;
	}
	/* the separators run from the tile's edge to 16px short of the card's */
	.row + .row::before {
		position: absolute;
		top: 0;
		right: 16px;
		left: 16px;
		height: 1px;
		background: rgb(255 255 255 / 0.1);
		content: '';
	}
	.tile {
		display: grid;
		flex: none;
		place-items: center;
		width: 32px;
		height: 32px;
		border-radius: 4px;
		background: #474541;
	}
	.logo {
		display: grid;
		place-items: center;
		width: 22.5px;
		height: 22.5px;
		color: #fff;
	}
	.logo :global(svg) {
		width: 100%;
		height: 100%;
	}
	/* Meta's mark in its blue on a white square, Liquid's in black on a white disc */
	.logo.square {
		width: 24px;
		height: 24px;
		padding: 3px;
		box-sizing: border-box;
		border-radius: 1px;
		background: #fff;
		color: #0668e1;
	}
	.logo.round {
		width: 23px;
		height: 23px;
		padding: 4px;
		box-sizing: border-box;
		border-radius: 50%;
		background: #fff;
		color: #000;
	}
	.info {
		display: grid;
		flex: 1;
		gap: 2px;
		min-width: 0;
	}
	.info b {
		overflow: hidden;
		font-weight: 600;
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.bar {
		margin: 0 7px;
		color: rgb(255 255 255 / 0.7);
		font-weight: 400;
	}
	.sizes {
		color: rgb(255 255 255 / 0.85);
		font-weight: 400;
	}
	.info small {
		overflow: hidden;
		color: rgb(255 255 255 / 0.85);
		font-size: 10.75px;
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.tags {
		display: flex;
		align-items: center;
		gap: 7px;
		color: rgb(255 255 255 / 0.85);
		font-size: 11px;
	}
	.tags i {
		display: inline-grid;
		place-items: center;
		height: 15px;
		padding: 0 4px;
		border-radius: 3px;
		background: #474541;
		font-style: normal;
	}
	.actions {
		display: flex;
		gap: 11.5px;
		color: rgb(255 255 255 / 0.8);
	}
	.row :global(.lucide-chevron-right) {
		flex: none;
		color: rgb(255 255 255 / 0.75);
	}
	.browse {
		display: flex;
		flex: none;
		align-items: center;
		gap: 8px;
		height: 28px;
		margin: 0;
		padding: 0 7.5px 0 8.5px;
		color: rgb(255 255 255 / 0.62);
		font-size: 11.5px;
	}
	.browse::before,
	.browse::after {
		flex: 1;
		height: 1px;
		background: rgb(255 255 255 / 0.1);
		content: '';
	}

	.search {
		display: flex;
		flex: none;
		gap: 4px;
		height: 32px;
		margin: 13.5px 14px 0 15px;
	}
	.hf {
		display: grid;
		flex: none;
		place-items: center;
		width: 32px;
		border-radius: 4px;
		background: #393937;
	}
	.hf img {
		width: 19px;
		height: 19px;
	}
	.field {
		display: flex;
		flex: 1;
		align-items: center;
		justify-content: space-between;
		padding: 0 14px 0 11.5px;
		border-radius: 4px;
		background: #343431;
		box-shadow: inset 0 -1px 0 #a4a4a2;
		color: rgb(255 255 255 / 0.78);
		font-size: 12.5px;
	}
	.status {
		display: flex;
		flex: 1;
		align-items: center;
		justify-content: center;
		gap: 7px;
		margin: 13px 15px 0;
		border-top: 1px solid rgb(255 255 255 / 0.1);
		font-size: 11px;
		font-weight: 600;
	}
	.status :global(svg) {
		color: #4d9be6;
	}
	.dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: #6dbe61;
	}
</style>
