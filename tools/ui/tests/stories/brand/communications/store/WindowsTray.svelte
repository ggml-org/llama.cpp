<!--
	Llama for Windows' tray flyout, drawn after the app (Morgan's build, November 2026) in Windows 11's dark theme: the
	icon with settings and quit, the Models card (the installed models with their tags and their folder, delete and run
	buttons, then "Browse more" over llama.app's catalog), a Hugging Face search under it, and the server's status and
	build. The catalog's makers carry their initials rather than their logos, as everywhere in communications. `card`
	draws the Models card alone, for a close-up.
-->
<script lang="ts">
	import { ICON } from '../posts/post';
	import {
		ChevronRight,
		Cpu,
		Eye,
		Folder,
		Play,
		Power,
		Search,
		Server,
		Settings,
		SquareArrowOutUpRight,
		Trash2
	} from '@lucide/svelte';

	interface Props {
		/** the Models card alone */
		card?: boolean;
		/** how many of the catalog's models show under "Browse more" */
		rows?: number;
	}

	let { card = false, rows = 5 }: Props = $props();

	// small and medium models from llama.app's catalog; the one running is highlighted
	const installed = [
		{ name: 'Gemma 4', running: true, size: '4.59 GB', tags: ['E4B', 'Q4_K_M'], vision: true },
		{ name: 'LFM 3 Omni', size: '1.8 GB', tags: ['2B', 'Q8_0'], vision: true }
	];
	const catalog = [
		{
			about: 'Liquid AI’s **fast local model** for reasoning and tool use.',
			initials: 'L',
			name: 'LFM 2.5',
			sizes: '24B-A2B'
		},
		{
			about: 'OpenAI’s open-weight family for **local reasoning and agents**.',
			initials: 'O',
			name: 'GPT-OSS',
			sizes: '20B · 120B'
		},
		{
			about: 'Alibaba’s multimodal **mid-sized models for coding and research**.',
			initials: 'Q',
			name: 'Qwen 3.8',
			sizes: '27B'
		},
		{
			about: 'Meta’s **local model to run on your laptop** powering always-on agents.',
			initials: 'M',
			name: 'Muse Glimmer',
			sizes: '30B'
		},
		{
			about: 'Google’s most capable open **agentic models for your laptops**.',
			initials: 'G',
			name: 'Gemma 4',
			sizes: 'E2B · E4B · 12B · 26B-A4B · 31B'
		}
	];
	// the catalog's descriptions mark their key phrase in bold
	const parts = (text: string) => text.split('**').map((t, i) => ({ bold: i % 2 === 1, t }));
</script>

{#snippet models()}
	<div class="card">
		<p class="card-title">Models <SquareArrowOutUpRight size={15} /></p>

		{#each installed as m (m.name)}
			<div class={['row', 'installed', { running: m.running }]}>
				<span class="tile"><Cpu size={20} /></span>

				<span class="info">
					<b>{m.name}</b>

					<span class="tags">
						{#each m.tags as tag (tag)}<i>{tag}</i>{/each}

						{#if m.vision}<i aria-label="vision"><Eye size={13} /></i>{/if}

						<span>{m.size}</span>
					</span>
				</span>

				<span class="actions"><Folder size={18} /><Trash2 size={18} /><Play size={18} /></span>
			</div>
		{/each}

		<p class="divider"><span>Browse more</span></p>

		{#each catalog.slice(0, rows) as m (m.name)}
			<div class="row">
				<span class="tile initials">{m.initials}</span>

				<span class="info">
					<b>{m.name} <span class="sizes">{m.sizes}</span></b>

					<small
						>{#each parts(m.about) as p, i (i)}{#if p.bold}<strong>{p.t}</strong
								>{:else}{p.t}{/if}{/each}</small
					>
				</span>

				<ChevronRight size={16} />
			</div>
		{/each}
	</div>
{/snippet}

{#if card}
	<div class="flyout only-card">{@render models()}</div>
{:else}
	<div class="flyout">
		<div class="head">
			<span class="icon">{ICON}</span>

			<span class="buttons"><Settings size={19} /><Power size={19} /></span>
		</div>

		{@render models()}

		<div class="search">
			<span class="hf">HF</span>

			<span class="field">Search models …<Search size={16} /></span>
		</div>

		<p class="status">
			<i class="dot"></i>0.14.1.0 · version 0.8.0 (build 12019)<Server size={15} />
		</p>
	</div>
{/if}

<style>
	/* Windows 11's dark flyout: a Mica-like dark grey, cards a step lighter, Segoe UI where there is one */
	.flyout {
		display: grid;
		gap: 12px;
		width: 400px;
		box-sizing: border-box;
		padding: 14px 8px 12px;
		border: 1px solid rgb(255 255 255 / 0.08);
		border-radius: 8px 8px 0 0;
		background: #202020;
		color: #fff;
		font:
			400 14px / 1.35 'Segoe UI Variable',
			'Segoe UI',
			var(--sans);
	}
	.flyout :global(p) {
		margin: 0;
	}
	.only-card {
		padding: 0;
		border: 0;
		background: none;
	}
	.head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0 6px 12px;
		border-bottom: 1px solid rgb(255 255 255 / 0.08);
	}
	.icon {
		font: 700 20px / 1 var(--sans);
	}
	.buttons {
		display: flex;
		gap: 16px;
		color: rgb(255 255 255 / 0.9);
	}
	.card {
		display: grid;
		padding: 12px 8px 8px;
		border: 1px solid rgb(255 255 255 / 0.06);
		border-radius: 8px;
		background: #2b2b2b;
	}
	.card-title {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 0 8px 10px;
		color: rgb(255 255 255 / 0.75);
		font-size: 15px;
	}
	.row {
		display: flex;
		align-items: center;
		gap: 12px;
		padding: 7px 8px;
		color: rgb(255 255 255 / 0.6);
	}
	.row + .row:not(.installed) {
		border-top: 1px solid rgb(255 255 255 / 0.07);
	}
	.running {
		border-radius: 6px;
		background: rgb(255 255 255 / 0.06);
	}
	.tile {
		display: grid;
		flex: none;
		place-items: center;
		width: 32px;
		height: 32px;
		border-radius: 6px;
		background: rgb(255 255 255 / 0.08);
		color: #fff;
	}
	.initials {
		font: 600 14px / 1 var(--sans);
	}
	.info {
		display: grid;
		flex: 1;
		gap: 3px;
		min-width: 0;
		color: #fff;
	}
	.info b {
		overflow: hidden;
		font-weight: 600;
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.sizes {
		margin-left: 4px;
		padding-left: 8px;
		border-left: 1px solid rgb(255 255 255 / 0.4);
		color: rgb(255 255 255 / 0.75);
		font-weight: 400;
	}
	.info small {
		overflow: hidden;
		color: rgb(255 255 255 / 0.75);
		font-size: 12.5px;
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.info strong {
		font-weight: 600;
	}
	.tags {
		display: flex;
		align-items: center;
		gap: 5px;
		color: rgb(255 255 255 / 0.8);
		font-size: 12.5px;
	}
	.tags i {
		display: inline-grid;
		place-items: center;
		height: 20px;
		padding: 0 6px;
		border-radius: 4px;
		background: rgb(255 255 255 / 0.1);
		font-style: normal;
	}
	.actions {
		display: flex;
		gap: 12px;
		color: rgb(255 255 255 / 0.75);
	}
	.divider {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 6px 0;
		color: rgb(255 255 255 / 0.6);
		font-size: 13px;
	}
	.divider::before,
	.divider::after {
		flex: 1;
		height: 1px;
		background: rgb(255 255 255 / 0.12);
		content: '';
	}
	.search {
		display: flex;
		gap: 6px;
	}
	.hf {
		display: grid;
		flex: none;
		place-items: center;
		width: 34px;
		border-radius: 6px;
		background: #2b2b2b;
		color: #ffd21e;
		font: 700 12px / 1 var(--sans);
	}
	.field {
		display: flex;
		flex: 1;
		align-items: center;
		justify-content: space-between;
		padding: 8px 12px;
		border-radius: 6px;
		border-bottom: 2px solid #4cc2ff;
		background: #1c1c1c;
		color: rgb(255 255 255 / 0.65);
	}
	.status {
		display: flex;
		align-items: center;
		gap: 8px;
		justify-content: center;
		padding-top: 10px;
		border-top: 1px solid rgb(255 255 255 / 0.08);
		color: rgb(255 255 255 / 0.85);
		font-size: 12.5px;
	}
	.status :global(svg) {
		color: #4cc2ff;
	}
	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: #3fb950;
	}
</style>
