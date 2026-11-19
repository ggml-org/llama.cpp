<!--
	The macOS Dock, for picturing an app icon in it: the translucent shelf, its apps on Apple's icon grid (the plate 824
	of each 1024 tile, as the files are drawn), a dot under the running ones, a separator before the trash, and the
	name label macOS shows over the app under the pointer. The other apps are neutral placeholders (plain plates with a
	Lucide glyph), so the one shown is the only real icon. Lay it on MockDesktop.
-->
<script lang="ts" module>
	import { Folder, Globe, Mail, MessageCircle, Music, Settings, Trash2 } from '@lucide/svelte';
	import type { Component } from 'svelte';

	export interface DockApp {
		/** a placeholder's plate colours, top to bottom, and its glyph */
		colors?: [string, string];
		glyph?: Component;
		/** an app's real icon: a full 1024 macOS tile, its plate on Apple's grid */
		icon?: string;
		name: string;
		running?: boolean;
	}

	// the neutral apps around the one shown: calm plates, a generic glyph each
	export const PLACEHOLDERS: DockApp[] = [
		{ colors: ['#5ea8ff', '#2f6fdc'], glyph: Folder, name: 'Files', running: true },
		{ colors: ['#7cc8ff', '#2d8fe0'], glyph: Globe, name: 'Browser', running: true },
		{ colors: ['#6fd28a', '#2fa35a'], glyph: MessageCircle, name: 'Messages' },
		{ colors: ['#8fb6ff', '#4d74e8'], glyph: Mail, name: 'Mail' },
		{ colors: ['#ff8a9a', '#e5435e'], glyph: Music, name: 'Music' },
		{ colors: ['#b8bcc4', '#7d828c'], glyph: Settings, name: 'Settings' }
	];
</script>

<script lang="ts">
	interface Props {
		/** the apps, left to right; the trash follows a separator */
		apps?: DockApp[];
		/** the app whose name label shows, as under the pointer */
		labelled?: string;
		mode?: 'dark' | 'light';
	}

	let { apps = PLACEHOLDERS, labelled, mode = 'light' }: Props = $props();
</script>

{#snippet tile(app: DockApp)}
	<span class="tile">
		{#if app.name === labelled}<span class="label">{app.name}</span>{/if}

		{#if app.icon}
			<img alt={app.name} src={app.icon} />
		{:else}
			{@const Glyph = app.glyph ?? Folder}
			<svg viewBox="0 0 1024 1024">
				<defs>
					<linearGradient id="plate-{app.name}" x1="0" x2="0" y1="0" y2="1">
						<stop offset="0" stop-color={app.colors?.[0] ?? '#d0d4da'} />
						<stop offset="1" stop-color={app.colors?.[1] ?? '#9aa0aa'} />
					</linearGradient>
				</defs>
				<rect fill="url(#plate-{app.name})" height="824" rx="185.4" width="824" x="100" y="100" />
			</svg>
			<span class="glyph"><Glyph strokeWidth={1.75} /></span>
		{/if}

		<span class={['dot', app.running && 'on']}></span>
	</span>
{/snippet}

<div aria-hidden="true" class={['dock', mode]}>
	{#each apps as app (app.name)}{@render tile(app)}{/each}

	<span class="separator"></span>

	{@render tile({ colors: ['#c4c8cf', '#8e939c'], glyph: Trash2, name: 'Trash' })}
</div>

<style>
	.dock {
		display: flex;
		align-items: flex-end;
		align-self: center;
		margin-bottom: 6px;
		padding: 4px 4px 2px;
		border: 1px solid rgb(255 255 255 / 0.55);
		border-radius: 22px;
		background: rgb(255 255 255 / 0.32);
		box-shadow:
			0 0 0 1px rgb(0 0 0 / 0.08),
			0 10px 30px -10px rgb(0 0 0 / 0.25);
		backdrop-filter: blur(24px) saturate(1.6);
		font:
			400 13px / 1 -apple-system,
			BlinkMacSystemFont,
			'SF Pro Text',
			system-ui,
			sans-serif;
	}
	.dark {
		border-color: rgb(255 255 255 / 0.14);
		background: rgb(40 40 46 / 0.4);
		box-shadow:
			0 0 0 1px rgb(0 0 0 / 0.45),
			0 10px 30px -10px rgb(0 0 0 / 0.6);
	}
	.tile {
		position: relative;
		display: grid;
		place-items: center;
		width: 60px;
		height: 66px;
		padding-bottom: 6px;
	}
	.tile img,
	.tile svg {
		grid-area: 1 / 1;
		width: 60px;
		height: 60px;
		filter: drop-shadow(0 2px 3px rgb(0 0 0 / 0.18));
	}
	/* positioned, so it paints over the plate, which its drop shadow lifts into a later layer */
	.glyph {
		position: relative;
		grid-area: 1 / 1;
		display: grid;
		place-items: center;
		color: white;
	}
	.glyph :global(svg) {
		width: 26px;
		height: 26px;
		filter: none;
	}
	.dot {
		position: absolute;
		bottom: 1px;
		width: 4px;
		height: 4px;
		border-radius: 50%;
	}
	.dot.on {
		background: rgb(0 0 0 / 0.6);
	}
	.dark .dot.on {
		background: rgb(255 255 255 / 0.75);
	}
	.separator {
		align-self: stretch;
		width: 1px;
		margin: 8px 6px 10px;
		background: rgb(0 0 0 / 0.16);
	}
	.dark .separator {
		background: rgb(255 255 255 / 0.18);
	}
	/* the name macOS shows over the app under the pointer */
	.label {
		position: absolute;
		bottom: calc(100% + 6px);
		left: 50%;
		translate: -50%;
		padding: 5px 10px;
		border: 1px solid rgb(255 255 255 / 0.6);
		border-radius: 8px;
		background: rgb(246 246 246 / 0.85);
		box-shadow: 0 4px 14px -4px rgb(0 0 0 / 0.25);
		color: rgb(0 0 0 / 0.85);
		white-space: nowrap;
		backdrop-filter: blur(20px);
	}
	.dark .label {
		border-color: rgb(255 255 255 / 0.12);
		background: rgb(44 44 48 / 0.85);
		color: rgb(255 255 255 / 0.92);
	}
</style>
