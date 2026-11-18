<!--
	How this Storybook is laid out, as one column in sidebar order: each section is a stop on a rail, with its icon, what
	it holds and the pages under it, as in the sidebar. Introduction is marked as where you are, in cinder.
-->
<script lang="ts">
	import { PATHS, SIZE } from '../brand/logo-showcase/lib/logo';
	import { STORYBOOK_SECTIONS } from './sections';
	import { BookOpen, Bot, LayoutGrid, MessageSquare } from '@lucide/svelte';

	const ICONS: Record<string, typeof BookOpen> = {
		Agents: Bot,
		Introduction: BookOpen,
		'Llama Chat': MessageSquare,
		'Llama UI': LayoutGrid
	};

	const sections = STORYBOOK_SECTIONS.map((section) => ({
		...section,
		here: section.name === 'Introduction',
		icon: ICONS[section.name]
	}));
</script>

<ol class="sections">
	{#each sections as section (section.name)}
		<li class="section">
			<span
				class={['glyph rounded-control bg-surface-lifted elevated-control', { here: section.here }]}
			>
				{#if section.icon}
					<section.icon aria-hidden="true" size={16} />
				{:else}
					<svg aria-hidden="true" class="icon" viewBox="0 0 {SIZE} {SIZE}">
						{#each PATHS as d (d)}<path {d} />{/each}
					</svg>
				{/if}
			</span>

			<div class="body card-surface rounded-card">
				<p class="head">
					<b>{section.name}</b>

					{#if section.here}<span class="you">You are here</span>{/if}
				</p>

				<p class="holds">{section.holds}</p>

				{#if section.pages.length}
					<ul class="pages">
						{#each section.pages as page (page)}
							<li class="page rounded-control">{page}</li>
						{/each}
					</ul>
				{/if}
			</div>
		</li>
	{/each}
</ol>

<style>
	.sections {
		position: relative;
		display: grid;
		gap: 8px;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	/* the rail the sections sit on, from the first glyph to the last */
	.sections::before {
		position: absolute;
		top: 24px;
		bottom: 24px;
		left: 17px;
		width: 1px;
		background: var(--line-default);
		content: '';
	}

	.section {
		display: grid;
		grid-template-columns: 36px minmax(0, 1fr);
		gap: 12px;
		align-items: start;
		margin: 0;
	}

	.glyph {
		position: relative;
		display: grid;
		place-items: center;
		width: 36px;
		height: 36px;
		margin-top: 8px;
	}

	.glyph.here {
		color: var(--art-cinder);
	}

	.icon {
		width: 16px;
		height: 16px;
		fill: currentColor;
	}

	.body {
		display: grid;
		gap: 4px;
		padding: 12px 14px;
	}

	p {
		margin: 0;
	}

	.head {
		display: flex;
		align-items: baseline;
		gap: 10px;
	}

	b {
		font-size: 15px;
		font-weight: 600;
	}

	.you {
		color: var(--art-cinder);
		font: 500 11px var(--mono);
		letter-spacing: 0.06em;
		text-transform: uppercase;
	}

	.holds {
		color: var(--figure-muted);
		font-size: 13px;
		line-height: 1.4;
	}

	.pages {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		margin: 6px 0 0;
		padding: 0;
		list-style: none;
	}

	.page {
		margin: 0;
		padding: 2px 8px;
		border: 1px solid var(--line-default);
		color: var(--figure-muted);
		font-size: 12px;
		line-height: 1.5;
	}
</style>
