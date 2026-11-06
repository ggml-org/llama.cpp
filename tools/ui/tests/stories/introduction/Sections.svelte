<!--
	How this Storybook is laid out: its sections in sidebar order, each with its icon and what it holds. Agents and
	Philosophy are paired as the rules and the reasons; the brand's tile carries the icon.
-->
<script lang="ts">
	import { PATHS, SIZE } from '../brand/logo-showcase/lib/logo';
	import { BookOpen, Bot, LayoutGrid, Lightbulb, Megaphone, Shapes } from '@lucide/svelte';

	const sections = [
		{ holds: 'This page', icon: BookOpen, name: 'Introduction' },
		{ holds: 'DESIGN.md and BRAND.md: the rules, for agents', icon: Bot, name: 'Agents' },
		{
			holds: 'Ideology, Product, Design: the reasons, for people',
			icon: Lightbulb,
			name: 'Philosophy'
		},
		{ holds: 'The logo, its files, the type and the colours', icon: null, name: 'Brand' },
		{
			holds: 'Posts, store screenshots, link previews, swag',
			icon: Megaphone,
			name: 'Communications'
		},
		{
			holds: 'Foundations, components, patterns, Llama Chat’s screens',
			icon: LayoutGrid,
			name: 'UI'
		},
		{ holds: 'Variations side by side, before one is chosen', icon: Shapes, name: 'Scrapbook' }
	];
</script>

<ol class="sections">
	{#each sections as section (section.name)}
		<li class="section card-surface rounded-card">
			<span class="glyph rounded-control bg-surface-lifted elevated-control">
				{#if section.icon}
					<section.icon aria-hidden="true" size={16} />
				{:else}
					<svg aria-hidden="true" class="icon" viewBox="0 0 {SIZE} {SIZE}">
						{#each PATHS as d (d)}<path {d} />{/each}
					</svg>
				{/if}
			</span>

			<b>{section.name}</b>

			<span class="holds">{section.holds}</span>
		</li>
	{/each}
</ol>

<style>
	.sections {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
		gap: 8px;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.section {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr);
		gap: 2px 12px;
		align-items: center;
		margin: 0;
		padding: 12px 14px;
	}

	.glyph {
		display: grid;
		grid-row: span 2;
		place-items: center;
		width: 32px;
		height: 32px;
	}

	.icon {
		width: 16px;
		height: 16px;
		fill: currentColor;
	}

	b {
		font-size: 15px;
		font-weight: 600;
	}

	.holds {
		color: var(--figure-muted);
		font-size: 13px;
		line-height: 1.4;
	}
</style>
