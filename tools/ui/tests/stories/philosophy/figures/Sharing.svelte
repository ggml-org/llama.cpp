<!--
	Shared, as a slideshow of four setups, from one person to the internet: a laptop running the Llama app, reached from
	a smartphone through Llama Connect; a home server on a small workstation for the family; llama-server on an office
	server for the people who work there; and a cluster of small workstations, two stacked, open to the internet. Each
	slide draws the machine that runs the model with a communications motif, dashed lines to the devices that use it,
	and how they connect. The tabs are Llama UI's own; the motifs read their inks from the --p-* variables, set here
	from the theme so they follow light and dark.
-->
<script lang="ts">
	import Laptop from '../../brand/communications/posts/motifs/Laptop.svelte';
	import Server from '../../brand/communications/posts/motifs/Server.svelte';
	import WorkstationSmall from '../../brand/communications/posts/motifs/WorkstationSmall.svelte';
	import { Globe, Laptop as LaptopIcon, Monitor, Smartphone, Tablet } from '@lucide/svelte';
	import * as Tabs from '$lib/components/ui/tabs';
	import type { Component } from 'svelte';

	interface Setup {
		clients: { icon: Component; name: string; note?: string }[];
		host: { count: number; machine: Component; name: string; note: string };
		id: string;
		link: string;
		line: string;
		name: string;
	}

	const setups: Setup[] = [
		{
			clients: [
				{ icon: Smartphone, name: 'Smartphone', note: 'Llama Chat, added to the Home Screen' }
			],
			host: {
				count: 1,
				machine: Laptop,
				name: 'Laptop',
				note: 'A MacBook with an M chip, or a laptop with an NVIDIA RTX Spark'
			},
			id: 'personal',
			line: 'The Llama app runs on your laptop; your smartphone opens Llama Chat as a web app and reaches it through Llama Connect.',
			link: 'Llama Connect',
			name: 'Your laptop, from your phone'
		},
		{
			clients: [
				{ icon: LaptopIcon, name: 'Laptops' },
				{ icon: Tablet, name: 'Tablets' },
				{ icon: Smartphone, name: 'Phones' }
			],
			host: {
				count: 1,
				machine: WorkstationSmall,
				name: 'Home server',
				note: 'A Mac Studio, a DGX Spark or a Ryzen AI Halo'
			},
			id: 'home',
			line: 'One small workstation at home runs bigger models than any phone, and the family connects from their own devices.',
			link: 'Home network',
			name: 'A home server for the family'
		},
		{
			clients: [
				{ icon: Monitor, name: 'Desktops' },
				{ icon: LaptopIcon, name: 'Laptops' }
			],
			host: { count: 1, machine: Server, name: 'Office server', note: 'llama-server' },
			id: 'office',
			line: 'llama-server runs on a server in the office; whoever runs it grants the people who work there access.',
			link: 'An API key each',
			name: 'A server for the office'
		},
		{
			clients: [{ icon: Globe, name: 'Anyone you let in', note: 'From anywhere' }],
			host: {
				count: 2,
				machine: WorkstationSmall,
				name: 'A cluster',
				note: 'Home servers or workstations, together'
			},
			id: 'internet',
			line: 'A home server or a cluster of workstations runs the largest open models, and is opened to the internet.',
			link: 'The internet',
			name: 'Open to the internet'
		}
	];

	let current = $state(setups[0].id);
</script>

<Tabs.Root bind:value={current} class="sharing">
	<Tabs.List aria-label="Setups">
		{#each setups as setup, i (setup.id)}
			<Tabs.Trigger value={setup.id}>
				<span class="n">0{i + 1}</span>

				<span class="tab-name">{setup.name}</span>
			</Tabs.Trigger>
		{/each}
	</Tabs.List>

	{#each setups as setup (setup.id)}
		<Tabs.Content value={setup.id}>
			<div class="slide">
				<div class="stage rounded-panel">
					<div class="host">
						<div class={['machines', { cluster: setup.host.count > 1 }]}>
							{#each Array(setup.host.count) as _, i (i)}
								<div class="motif"><setup.host.machine /></div>
							{/each}
						</div>

						<b>{setup.host.name}</b>

						<span>{setup.host.note}</span>
					</div>

					<div class="link">
						<span aria-hidden="true" class="line"></span>

						<span class="via">{setup.link}</span>
					</div>

					<ul class="clients">
						{#each setup.clients as client (client.name)}
							<li class="client card-surface rounded-card">
								<client.icon aria-hidden="true" size={22} />

								<b>{client.name}</b>

								{#if client.note}<span>{client.note}</span>{/if}
							</li>
						{/each}
					</ul>
				</div>

				<p class="caption"><b>{setup.name}.</b> {setup.line}</p>
			</div>
		</Tabs.Content>
	{/each}
</Tabs.Root>

<style>
	/* a grid item in the figure, one column as wide as the frame: the slides fit it and the tabs scroll inside it */
	:global(.sharing) {
		container-type: inline-size;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
		--p-base: var(--surface-page);
		--p-ink: var(--fg-default);
		--p-muted: var(--figure-muted);
		--p-inset: color-mix(in oklab, var(--fg-default) 6%, transparent);
		display: grid;
		gap: 12px;
	}

	:global(.sharing [role='tablist']) {
		justify-self: start;
		max-width: 100%;
	}

	.n {
		color: var(--figure-muted);
		font: 500 11px var(--mono);
	}

	.slide {
		display: grid;
		gap: 10px;
	}

	.stage {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(6rem, 0.7fr) minmax(0, 1fr);
		align-items: center;
		gap: 12px;
		min-height: 260px;
		padding: 20px;
		background: var(--surface-page);
	}

	.host {
		display: grid;
		justify-items: center;
		gap: 2px;
		text-align: center;
	}

	.machines {
		display: flex;
		justify-content: center;
		width: 100%;
	}

	.motif {
		width: min(100%, 200px);
		aspect-ratio: 4 / 3;
	}

	/* two machines stacked, as one cluster: both in one cell, the second raised by a box's height and drawn over the
	first */
	.cluster {
		display: grid;
		justify-items: center;
		padding-top: 34px;
	}

	.cluster .motif {
		grid-area: 1 / 1;
		width: min(100%, 170px);
	}

	.cluster .motif + .motif {
		z-index: 1;
		transform: translateY(-34px);
	}

	b {
		font-size: 15px;
		font-weight: 600;
	}

	.host span,
	.client span {
		color: var(--figure-muted);
		font-size: 12px;
		line-height: 1.4;
	}

	.link {
		display: grid;
		justify-items: center;
		gap: 8px;
	}

	.line {
		width: 100%;
		border-top: 2px dashed var(--line-default);
	}

	.via {
		color: var(--art-cinder);
		font: 500 12px var(--mono);
		text-align: center;
	}

	.clients {
		display: grid;
		gap: 8px;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.client {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr);
		gap: 0 10px;
		align-items: center;
		margin: 0;
		padding: 10px 12px;
	}

	.client :global(svg) {
		grid-row: span 2;
	}

	.caption {
		margin: 0;
		color: var(--figure-muted);
		font-size: 14px;
		line-height: 1.5;
	}

	.caption b {
		color: var(--fg-default);
		font-size: inherit;
	}

	/* the tabs stay one row of names as the frame narrows: smaller and tighter first (the row takes 816px at 14px, 675
	at 12), then two columns of names inside the same track, then one on a phone */
	@container (width < 840px) {
		:global(.sharing [role='tab']) {
			padding-inline: 8px;
			font-size: 12px;
			letter-spacing: -0.01em;
		}
	}

	@container (width < 690px) {
		:global(.sharing [role='tablist']) {
			display: grid;
			grid-template-columns: repeat(2, minmax(0, 1fr));
			justify-self: stretch;
			height: auto;
		}

		:global(.sharing [role='tab']) {
			justify-content: start;
			min-height: var(--height-control-md);
		}
	}

	@container (width < 420px) {
		:global(.sharing [role='tablist']) {
			grid-template-columns: minmax(0, 1fr);
		}
	}

	@media (width < 720px) {
		.stage {
			grid-template-columns: minmax(0, 1fr);
			min-height: 0;
		}

		.line {
			width: 0;
			height: 28px;
			border-top: 0;
			border-left: 2px dashed var(--line-default);
		}
	}
</style>
