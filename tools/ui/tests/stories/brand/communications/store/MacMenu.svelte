<!--
	Llama for Mac's menu (0.47), rebuilt from its source (ggml-org/Llama-macOS, Llama/Menu: Layout, Theme, HeaderView,
	ModelItemView, CatalogItemView, BrowseModelsRow, FooterView) in light mode, every size in points as the app sets it:
	300 wide, rows inset 5 then 8, 40-point model rows with a 28-point round icon, 13 and 11-point system type at 85,
	45 and 32% black. The header with the server's dot and address (and, with `network`, the QR code for phones), the
	installed models with their chips, Recommended for your Mac, Browse models, and the version with Settings and
	Quit. The models are small and medium ones from llama.app's catalog; their logos are the app's own ModelLogos.
	`installed` and `recommended` replace the rows (none installed is the first launch, with a grey dot), and
	`downloading` shows the first recommended row's download in progress.
-->
<script lang="ts" module>
	export interface MacMenuRow {
		active?: boolean;
		logo: string;
		name: string;
		params?: string;
		quant?: string;
		subtitle: string;
		vision?: boolean;
	}
</script>

<script lang="ts">
	import qr from './assets/qr-network.svg?raw';
	import gemma from './logos/gemma.svg?raw';
	import gpt from './logos/gpt.svg?raw';
	import liquid from './logos/liquid.svg?raw';
	import qwen from './logos/qwen.svg?raw';
	import {
		ArrowDown,
		ArrowUpRight,
		ChevronRight,
		Copy,
		Glasses,
		LayoutGrid,
		QrCode
	} from '@lucide/svelte';

	type Row = MacMenuRow;

	interface Props {
		/** the first recommended row's download, from 0 to 1 */
		downloading?: number;
		/** the installed models, in place of the default ones; none is the first launch */
		installed?: Row[];
		/** the server reachable from other devices: its network address, and the QR code open */
		network?: boolean;
		/** the models recommended for this Mac, in place of the default ones */
		recommended?: Row[];
	}

	const INSTALLED: Row[] = [
		{
			active: true,
			logo: gemma,
			name: 'gemma-4',
			params: 'E4B',
			quant: 'Q4_K_M',
			subtitle: '4.59 GB  ∣  128k ctx',
			vision: true
		},
		{
			logo: gpt,
			name: 'gpt-oss',
			params: '20B',
			quant: 'MXFP4',
			subtitle: '12.1 GB  ∣  128k ctx'
		},
		{
			logo: liquid,
			name: 'LFM3-Omni',
			params: '2B',
			quant: 'Q8_0',
			subtitle: '1.8 GB  ∣  32k ctx',
			vision: true
		}
	];
	const RECOMMENDED: Row[] = [
		{
			logo: liquid,
			name: 'LFM2.5',
			params: '24B-A2B',
			quant: 'Q4_K_M',
			subtitle: 'Good first model  ∣  14.5 GB'
		},
		{
			logo: qwen,
			name: 'Qwen3.8',
			params: '27B',
			quant: 'Q4_K_M',
			subtitle: '16.8 GB',
			vision: true
		}
	];
	let {
		downloading,
		installed = INSTALLED,
		network = false,
		recommended = RECOMMENDED
	}: Props = $props();

	const host = $derived(network ? '100.101.42.7:9931' : 'localhost:9931');
</script>

{#snippet title(row: Row)}
	<span class="title">
		{row.name}{#if row.params}<i class="chip squared">{row.params}</i>{/if}{#if row.quant}<i
				class="chip rounded">{row.quant}</i
			>{/if}{#if row.vision}<Glasses class="vision" size={13} strokeWidth={1.75} />{/if}
	</span>
{/snippet}

{#snippet modelRow(row: Row, trailing: 'chevron' | 'install', progress?: number)}
	<div class="row model">
		<span class={['icon', { active: row.active }]}>{@html row.logo}</span>

		<span class="text">
			{@render title(row)}

			<span class="subtitle">{row.subtitle}</span>
		</span>

		{#if trailing === 'chevron'}
			<ChevronRight class="trailing" size={14} strokeWidth={2} />
		{:else if progress !== undefined}
			<span class="percent">{Math.round(progress * 100)}%</span>

			<i class="progress"><i style:width="{progress * 100}%"></i></i>
		{:else}
			<ArrowDown class="trailing" size={14} strokeWidth={2} />
		{/if}
	</div>
{/snippet}

<div class="menu">
	<div class="row header">
		<p class="name">Llama <i class={['dot', { idle: installed.length === 0 }]}></i></p>

		<p class="actions">
			<span>{host}</span>

			<Copy class="glyph" size={11} strokeWidth={2} />

			{#if network}<QrCode class="glyph qr" size={11} strokeWidth={2} />{/if}

			{#if installed.length}<span class="open">Open chat</span>{/if}
		</p>

		{#if network}<span class="qrcode">{@html qr}</span>{/if}
	</div>

	<i class="separator"></i>

	<p class="row section">Installed <span class="link">models</span></p>

	{#each installed as row (row.name)}{@render modelRow(row, 'chevron')}{:else}<p class="row none">
			None yet
		</p>{/each}

	<i class="separator"></i>

	<p class="row section">Recommended for your Mac</p>

	{#each recommended as row, i (row.name)}{@render modelRow(
			row,
			'install',
			i === 0 ? downloading : undefined
		)}{/each}

	<i class="separator"></i>

	<div class="row model">
		<span class="icon"><LayoutGrid size={16} strokeWidth={1.75} /></span>

		<span class="text">
			<span class="title">Browse models</span>

			<span class="subtitle">Full catalog on the web</span>
		</span>

		<ArrowUpRight class="trailing" size={14} strokeWidth={2} />
	</div>

	<i class="separator"></i>

	<div class="row footer">
		<span>0.47.0<span class="tertiary"> · llama.cpp b12110</span></span>

		<span class="buttons"><span>Settings</span><span>Quit</span></span>
	</div>
</div>

<style>
	/* the menu: macOS's light menu material, 300 points wide */
	.menu {
		--primary: rgb(0 0 0 / 0.85);
		--secondary: rgb(0 0 0 / 0.45);
		--tertiary: rgb(0 0 0 / 0.32);
		--subtle: rgb(0 0 0 / 0.06);
		--border: rgb(0 0 0 / 0.15);
		--separator: rgb(0 0 0 / 0.1);
		--link: #0068da;

		display: flex;
		flex-direction: column;
		width: 300px;
		box-sizing: border-box;
		padding: 5px 0;
		border: 0.5px solid rgb(0 0 0 / 0.12);
		border-radius: 10px;
		background: rgb(246 246 246 / 0.9);
		color: var(--primary);
		font:
			400 13px / 1.25 -apple-system,
			BlinkMacSystemFont,
			'SF Pro Text',
			system-ui,
			sans-serif;
		box-shadow:
			0 10px 30px rgb(0 0 0 / 0.18),
			0 0 0 0.5px rgb(0 0 0 / 0.06);
		backdrop-filter: blur(30px) saturate(1.6);
	}
	.menu p {
		margin: 0;
	}
	/* every row: inset 5 from the menu, then 8 more to its content, 4 above and below */
	.row {
		margin: 0 5px;
		padding: 4px 8px;
	}
	.header {
		display: grid;
		gap: 4px;
	}
	.name {
		display: flex;
		align-items: baseline;
		gap: 6px;
		font-weight: 600;
	}
	.dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: #34c759;
		transform: translateY(-1px);
	}
	.actions {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 11px;
	}
	.actions :global(.glyph) {
		color: var(--secondary);
	}
	.actions :global(.qr) {
		margin-left: 4px;
	}
	.open {
		margin-left: auto;
		color: var(--link);
	}
	.qrcode {
		display: block;
		width: 90px;
		height: 90px;
		margin-top: 6px;
		color: rgb(0 0 0 / 0.88);
	}
	.qrcode :global(svg) {
		display: block;
		width: 100%;
		height: 100%;
	}
	.dot.idle {
		background: rgb(0 0 0 / 0.25);
	}
	.none {
		color: var(--secondary);
		font-size: 11px;
	}
	.percent {
		color: var(--secondary);
		font-size: 11px;
		font-variant-numeric: tabular-nums;
	}
	.progress {
		position: absolute;
		right: 8px;
		bottom: 0;
		left: 44px;
		height: 3px;
		overflow: hidden;
		border-radius: 2px;
		background: var(--subtle);
	}
	.progress i {
		display: block;
		height: 100%;
		background: var(--link);
	}
	.separator {
		display: block;
		height: 1px;
		margin: 5.5px 13px;
		background: var(--separator);
	}
	.section {
		font-size: 11px;
	}
	.link {
		margin-left: 2px;
		color: var(--link);
	}
	.model {
		position: relative;
		display: flex;
		align-items: center;
		gap: 6px;
		height: 40px;
		box-sizing: border-box;
	}
	/* a model's icon: its maker's logo in a 28-point circle, in the accent when it is loaded */
	.icon {
		display: grid;
		flex: none;
		place-items: center;
		width: 28px;
		height: 28px;
		border-radius: 50%;
		background: var(--subtle);
		color: rgb(0 0 0 / 0.65);
	}
	.icon :global(svg) {
		width: 16px;
		height: 16px;
	}
	.icon.active {
		background: #007aff;
		color: #fff;
	}
	.text {
		display: grid;
		flex: 1;
		gap: 2px;
		min-width: 0;
	}
	.title {
		display: flex;
		align-items: center;
		gap: 4px;
		white-space: nowrap;
	}
	/* the chips, 14 points high in 9-point medium text: the size outlined, the quant filled */
	.chip {
		display: inline-grid;
		place-items: center;
		height: 14px;
		padding: 0 5px;
		box-sizing: border-box;
		color: var(--secondary);
		font:
			500 9px / 1 -apple-system,
			BlinkMacSystemFont,
			'SF Pro Text',
			system-ui,
			sans-serif;
		font-style: normal;
		letter-spacing: 0.01em;
	}
	.squared {
		border: 1px solid var(--border);
		border-radius: 4px;
	}
	.rounded {
		border-radius: 7px;
		background: var(--subtle);
	}
	.title :global(.vision) {
		color: var(--tertiary);
	}
	.subtitle {
		overflow: hidden;
		color: var(--secondary);
		font-size: 11px;
		white-space: pre;
		text-overflow: ellipsis;
	}
	.row :global(.trailing) {
		flex: none;
		color: rgb(0 0 0 / 0.26);
	}
	.footer {
		display: flex;
		align-items: center;
		justify-content: space-between;
		height: 28px;
		padding-block: 0;
		box-sizing: border-box;
		font-size: 11px;
	}
	.tertiary {
		color: var(--secondary);
	}
	.buttons {
		display: flex;
		gap: 5px;
	}
	.buttons span {
		padding: 2px 4px;
		border: 1px solid var(--border);
		border-radius: 5px;
		color: var(--secondary);
	}
</style>
