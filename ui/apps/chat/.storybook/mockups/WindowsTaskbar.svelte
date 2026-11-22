<!--
	A Windows 11 taskbar, for picturing an app on it: the full-width bar in its translucent material, the start button,
	the search box and the pinned apps centred (a short underline under the running ones, a longer one in the accent
	colour under the active one), and the tray on the right: the hidden icons' chevron, the app's tray icon, network,
	sound and battery, and the two-line clock. The other apps are neutral placeholders (plain tiles with a Lucide
	glyph), so the one shown is the only real icon. Set in Segoe UI where there is one. Lay it on MockDesktop.
-->
<script lang="ts" module>
	import {
		BatteryCharging,
		ChevronUp,
		Folder,
		Globe,
		Mail,
		Search,
		Settings,
		Volume2,
		Wifi
	} from '@lucide/svelte';
	import type { Component } from 'svelte';

	export interface TaskbarApp {
		/** the active app: its window is in front */
		active?: boolean;
		/** a placeholder's tile colour and glyph */
		color?: string;
		glyph?: Component;
		/** an app's real taskbar icon */
		icon?: string;
		name: string;
		running?: boolean;
	}

	export const PLACEHOLDERS: TaskbarApp[] = [
		{ color: '#f2b84b', glyph: Folder, name: 'Files', running: true },
		{ color: '#3a8ee6', glyph: Globe, name: 'Browser', running: true },
		{ color: '#4b7bec', glyph: Mail, name: 'Mail' },
		{ color: '#7d828c', glyph: Settings, name: 'Settings' }
	];
</script>

<script lang="ts">
	interface Props {
		/** the pinned apps, after the start button and the search box */
		apps?: TaskbarApp[];
		clock?: [string, string];
		mode?: 'dark' | 'light';
		/** the app's tray icon: a glyph's paths and viewBox, drawn in the taskbar's ink */
		trayIcon?: { paths: string[]; viewBox: string };
	}

	let {
		apps = PLACEHOLDERS,
		clock = ['10:24 AM', '9/23/2026'],
		mode = 'light',
		trayIcon
	}: Props = $props();
</script>

<div aria-hidden="true" class={['taskbar', mode]}>
	<div class="centre">
		<!-- the start button: a plain four-pane glyph -->
		<span class="button">
			<svg class="start" viewBox="0 0 24 24">
				<rect height="10.5" rx="1.5" width="10.5" x="1.5" y="1.5" />
				<rect height="10.5" rx="1.5" width="10.5" x="12.75" y="1.5" />
				<rect height="10.5" rx="1.5" width="10.5" x="1.5" y="12.75" />
				<rect height="10.5" rx="1.5" width="10.5" x="12.75" y="12.75" />
			</svg>
		</span>

		<span class="search"><Search size={16} strokeWidth={1.75} />Search</span>

		{#each apps as app (app.name)}
			{@const Glyph = app.glyph ?? Folder}
			<span class={['button', app.active && 'active']}>
				{#if app.icon}
					<img alt={app.name} src={app.icon} />
				{:else}
					<span style:background={app.color} class="tile"><Glyph size={15} strokeWidth={2} /></span>
				{/if}

				{#if app.running || app.active}<span class="underline"></span>{/if}
			</span>
		{/each}
	</div>

	<div class="tray">
		<span class="tray-button"><ChevronUp size={16} strokeWidth={1.75} /></span>

		{#if trayIcon}
			<span class="tray-button">
				<svg class="tray-icon" fill="currentColor" viewBox={trayIcon.viewBox}>
					{#each trayIcon.paths as d (d)}<path {d} />{/each}
				</svg>
			</span>
		{/if}

		<span class="tray-button system">
			<Wifi size={16} strokeWidth={1.75} />
			<Volume2 size={16} strokeWidth={1.75} />
			<BatteryCharging size={16} strokeWidth={1.75} />
		</span>

		<span class="clock"><span>{clock[0]}</span><span>{clock[1]}</span></span>
	</div>
</div>

<style>
	.taskbar {
		position: relative;
		display: flex;
		align-items: center;
		justify-content: flex-end;
		height: 48px;
		padding: 0 8px;
		border-top: 1px solid rgb(0 0 0 / 0.08);
		background: rgb(238 242 247 / 0.82);
		backdrop-filter: blur(30px) saturate(1.4);
		color: rgb(0 0 0 / 0.9);
		font:
			400 12px / 1.35 'Segoe UI Variable Text',
			'Segoe UI',
			system-ui,
			sans-serif;
	}
	.dark {
		border-top-color: rgb(255 255 255 / 0.08);
		background: rgb(28 32 40 / 0.82);
		color: rgb(255 255 255 / 0.92);
	}
	.centre {
		position: absolute;
		left: 50%;
		display: flex;
		align-items: center;
		gap: 4px;
		translate: -50%;
	}
	.button {
		position: relative;
		display: grid;
		place-items: center;
		width: 40px;
		height: 40px;
		border-radius: 6px;
	}
	.active {
		background: rgb(255 255 255 / 0.6);
		box-shadow: inset 0 0 0 1px rgb(0 0 0 / 0.05);
	}
	.dark .active {
		background: rgb(255 255 255 / 0.08);
		box-shadow: inset 0 0 0 1px rgb(255 255 255 / 0.05);
	}
	.start {
		width: 22px;
		height: 22px;
		fill: #2f7de1;
	}
	.dark .start {
		fill: #5aa6ff;
	}
	.search {
		display: flex;
		align-items: center;
		gap: 8px;
		width: 168px;
		height: 32px;
		margin: 0 4px;
		padding: 0 12px;
		border: 1px solid rgb(0 0 0 / 0.06);
		border-radius: 16px;
		background: rgb(255 255 255 / 0.75);
		color: rgb(0 0 0 / 0.55);
	}
	.dark .search {
		border-color: rgb(255 255 255 / 0.06);
		background: rgb(255 255 255 / 0.06);
		color: rgb(255 255 255 / 0.6);
	}
	.button img,
	.tile {
		width: 24px;
		height: 24px;
		border-radius: 5px;
	}
	.tile {
		display: grid;
		place-items: center;
		color: white;
	}
	.underline {
		position: absolute;
		bottom: 1px;
		width: 6px;
		height: 3px;
		border-radius: 2px;
		background: rgb(0 0 0 / 0.4);
	}
	.dark .underline {
		background: rgb(255 255 255 / 0.5);
	}
	.active .underline {
		width: 16px;
		background: #2f7de1;
	}
	.dark .active .underline {
		background: #5aa6ff;
	}
	.tray {
		display: flex;
		align-items: center;
		gap: 2px;
	}
	.tray-button {
		display: flex;
		align-items: center;
		gap: 10px;
		height: 40px;
		padding: 0 8px;
		border-radius: 6px;
	}
	.tray-icon {
		width: 16px;
		height: 16px;
	}
	.clock {
		display: flex;
		flex-direction: column;
		align-items: flex-end;
		padding: 0 8px;
	}
</style>
