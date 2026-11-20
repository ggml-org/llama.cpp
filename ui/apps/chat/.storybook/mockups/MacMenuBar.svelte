<!--
	A macOS menu bar, for picturing a menu bar app: as in current macOS, no background of its own, the text and icons
	straight on the wallpaper (lay it on MockDesktop). The front app's menus on the left, the status items on the right:
	the app's own (its template icon, drawn in the bar's ink, pressed when its menu is open), then the system's (real SF
	Symbols, see MacStatusIcon) and the clock. Set in the system font, so on a Mac it is San Francisco.
-->
<script lang="ts">
	import MacStatusIcon, { type MacStatusIconName } from './MacStatusIcon.svelte';

	interface Props {
		/** the front app's name, then its menus */
		app?: string;
		clock?: string;
		menus?: string[];
		mode?: 'dark' | 'light';
		/** the app's status item is pressed, as when its menu is open */
		pressed?: boolean;
		/** the app's status item: a template icon's paths and viewBox (the Llama app's MenuIcon by default) */
		statusIcon?: { paths: string[]; viewBox: string };
	}

	// the Llama app's menu bar icon, from ggml-org/Llama-macOS (Llama/Assets.xcassets/MenuIcon.imageset/MenuIcon.svg)
	const LLAMA_MENU_ICON = {
		paths: [
			'M15 10.1467L12.7658 14.02L12.7632 14.0183C12.3692 14.6283 11.6912 15 10.9602 15H3.8L6.6 10.1467H15Z',
			'M6.6 10.1467H1L5.65738 2.07393C6.04075 1.40941 6.74959 1 7.51677 1H11.8769L6.6 10.1467Z'
		],
		viewBox: '0 0 16 16'
	};

	const SYSTEM: MacStatusIconName[] = ['sound', 'wifi', 'battery', 'spotlight', 'controlCenter'];

	let {
		app = 'Finder',
		clock = 'Wed Sep 23  10:24',
		menus = ['File', 'Edit', 'View', 'Go', 'Window', 'Help'],
		mode = 'light',
		pressed = true,
		statusIcon = LLAMA_MENU_ICON
	}: Props = $props();
</script>

<div aria-hidden="true" class={['bar', mode]}>
	<div class="menus">
		<span class="app">{app}</span>

		{#each menus as menu (menu)}<span>{menu}</span>{/each}
	</div>

	<div class="status">
		<span class={['item', pressed && 'pressed']}>
			<svg class="status-icon" fill="currentColor" viewBox={statusIcon.viewBox}>
				{#each statusIcon.paths as d (d)}<path {d} />{/each}
			</svg>
		</span>

		{#each SYSTEM as name (name)}<span class="item"><MacStatusIcon {name} /></span>{/each}

		<span class="clock">{clock}</span>
	</div>
</div>

<style>
	.bar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		height: 28px;
		padding: 0 10px 0 16px;
		font:
			500 13px / 1 -apple-system,
			BlinkMacSystemFont,
			'SF Pro Text',
			system-ui,
			sans-serif;
		white-space: pre;
		color: rgb(0 0 0 / 0.85);
		/* the faint scrim current macOS lays under the bar for legibility */
		background: linear-gradient(rgb(255 255 255 / 0.18), rgb(255 255 255 / 0));
	}
	.dark {
		color: rgb(255 255 255 / 0.92);
		background: linear-gradient(rgb(0 0 0 / 0.18), rgb(0 0 0 / 0));
	}
	.menus {
		display: flex;
		gap: 20px;
		overflow: hidden;
	}
	.app {
		font-weight: 700;
	}
	.status {
		display: flex;
		align-items: center;
		gap: 4px;
	}
	.item {
		display: flex;
		align-items: center;
		height: 22px;
		padding: 0 7px;
		border-radius: 6px;
	}
	.pressed {
		background: rgb(0 0 0 / 0.12);
	}
	.dark .pressed {
		background: rgb(255 255 255 / 0.22);
	}
	.status-icon {
		width: 16px;
		height: 16px;
	}
	.clock {
		padding: 0 4px 0 7px;
	}
</style>
