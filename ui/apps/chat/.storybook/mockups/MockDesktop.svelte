<!--
	A crop of a desktop for the OS mockups, laid out at a screen's size (at least width, filling a wider frame) and
	scaled down to fit a narrower one, like a screenshot: the wallpaper llama.app's site uses for its menu bar picture
	(ggml-org/llama-pages), a clear daytime sky in light, the same sky at night in dark, with the mockup laid on it.
	align puts the mockup at the top (a menu bar), at the bottom (a Dock, a taskbar), in the middle, or splits it, the
	first at the top and the last at the bottom (a menu bar over a Dock).
-->
<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		align?: 'between' | 'bottom' | 'center' | 'top';
		children: Snippet;
		/** the screen's size the mockup is laid out at, in px: it scales down, as a screenshot would, to fit */
		height?: number;
		mode?: 'dark' | 'light';
		width?: number;
	}

	let { align = 'center', children, height = 192, mode = 'light', width = 960 }: Props = $props();

	// the frame's width: wider than the screen, the screen widens to fill it; narrower, it scales down
	let fit = $state(0);
	const scale = $derived(fit ? Math.min(1, fit / width) : 1);
	const stage = $derived(Math.max(width, fit));
</script>

<div style:height="{height * scale}px" class="frame" bind:clientWidth={fit}>
	<div
		style:width="{stage}px"
		style:height="{height}px"
		style:scale
		class={['desktop', mode, align]}
	>
		{@render children()}
	</div>
</div>

<style>
	.frame {
		overflow: hidden;
		border-radius: 14px;
		box-shadow: 0 0 0 1px rgb(0 0 0 / 0.08);
	}
	.desktop {
		display: flex;
		flex-direction: column;
		transform-origin: 0 0;
		color-scheme: light;
	}
	.light {
		background:
			radial-gradient(ellipse at 15% 10%, #dcecfb 0%, transparent 55%),
			radial-gradient(ellipse at 90% 95%, #e8f2fc 0%, transparent 50%),
			linear-gradient(160deg, #c4ddf6 0%, #d0e4f7 55%, #dfecf9 100%);
	}
	.dark {
		background:
			radial-gradient(ellipse at 15% 10%, #22385c 0%, transparent 55%),
			radial-gradient(ellipse at 90% 95%, #1e3752 0%, transparent 50%),
			linear-gradient(160deg, #152238 0%, #172a42 55%, #1a3049 100%);
		color-scheme: dark;
	}
	.top {
		justify-content: flex-start;
	}
	.center {
		justify-content: center;
	}
	.bottom {
		justify-content: flex-end;
	}
	.between {
		justify-content: space-between;
	}
</style>
