<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import logoMark from '$lib/assets/logo.svg?raw';
	import { FAVICON_COLORS, PWA_ASSET_GENERATOR, THEME_COLORS } from '$lib/constants';

	// the icon and splash files in static/ are build output (npm run build-pwa-assets), so this
	// draws them from their source with the same proportions: logo.svg in the favicon colours,
	// padded as pwa-assets.config.ts and scripts/make-icons-circular.js pad it
	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Brand/App icons'
	});

	// @vite-pwa/assets-generator's default padding for the maskable and Apple touch icons
	const GENERATOR_ICON_PADDING = 0.3;
	// make-icons-circular.js scales the maskable icon to 85% of the circle
	const CIRCULAR_ICON_SCALE = 0.85;

	// every output starts from favicon.svg, the logo already shrunk by the favicon padding
	const favicon = 1 - PWA_ASSET_GENERATOR.FAVICON_PADDING;
	const squareIcon = (1 - GENERATOR_ICON_PADDING) * favicon;
	const circularIcon = CIRCULAR_ICON_SCALE * squareIcon;
	// the splash logo is fitted into this share of the screen's shorter side
	const splashLogo = (1 - PWA_ASSET_GENERATOR.SPLASH_PADDING) * favicon;

	const lightMark = logoMark.replaceAll('currentColor', FAVICON_COLORS.LIGHT);
	const darkMark = logoMark.replaceAll('currentColor', FAVICON_COLORS.DARK);

	const percent = (fraction: number) => `${Math.round(fraction * 100)}%`;

	const icons = [
		{
			background: THEME_COLORS.BACKGROUND_LIGHT,
			files: 'favicon.svg, favicon.ico',
			label: 'Favicon',
			mark: lightMark,
			note: `logo ${percent(favicon)}`,
			scale: favicon,
			shape: ''
		},
		{
			background: THEME_COLORS.BACKGROUND_DARK,
			files: 'favicon-dark.svg, favicon-dark.ico',
			label: 'Favicon dark',
			mark: darkMark,
			note: `logo ${percent(favicon)}`,
			scale: favicon,
			shape: ''
		},
		{
			background: THEME_COLORS.BACKGROUND_LIGHT,
			files: 'pwa-64x64, 192x192, 512x512.png',
			label: 'PWA',
			mark: lightMark,
			note: `logo ${percent(circularIcon)}`,
			scale: circularIcon,
			shape: 'rounded-full'
		},
		{
			background: THEME_COLORS.BACKGROUND_LIGHT,
			files: 'maskable-icon-512x512.png',
			label: 'Maskable',
			mark: lightMark,
			note: `logo ${percent(squareIcon)}`,
			safeZone: true,
			scale: squareIcon,
			shape: ''
		},
		{
			background: THEME_COLORS.BACKGROUND_LIGHT,
			files: 'apple-touch-icon-180x180.png',
			label: 'Apple touch',
			mark: lightMark,
			note: `logo ${percent(squareIcon)}`,
			scale: squareIcon,
			shape: 'rounded-[22.5%]'
		}
	];

	const splashes = [
		{
			background: THEME_COLORS.BACKGROUND_LIGHT,
			files: 'apple-splash-portrait-*.png',
			height: 2556,
			label: 'iPhone',
			mark: lightMark,
			note: '1179 × 2556',
			width: 1179
		},
		{
			background: THEME_COLORS.BACKGROUND_DARK,
			files: 'apple-splash-portrait-dark-*.png',
			height: 2556,
			label: 'iPhone dark',
			mark: darkMark,
			note: '1179 × 2556',
			width: 1179
		},
		{
			background: THEME_COLORS.BACKGROUND_LIGHT,
			files: 'apple-splash-landscape-*.png',
			height: 2048,
			label: 'iPad landscape',
			mark: lightMark,
			note: '2732 × 2048',
			width: 2732
		}
	];
</script>

<Story asChild name="Icons">
	<StateGallery columns={5} minItemWidth="9rem">
		{#each icons as icon (icon.label)}
			<StateGalleryItem label={icon.label} note={icon.note}>
				<div
					style:background-color={icon.background}
					class="relative flex size-24 items-center justify-center overflow-hidden ring-1 ring-line-default {icon.shape}"
				>
					<div style:width={percent(icon.scale)} class="mark aspect-square">
						{@html icon.mark}
					</div>

					{#if icon.safeZone}
						<!-- launchers may crop a maskable icon down to this 80% circle -->
						<div
							class="pointer-events-none absolute inset-[10%] rounded-full border border-dashed border-(--fg-danger)"
						></div>
					{/if}
				</div>

				<p class="mt-2 text-xs text-fg-muted">{icon.files}</p>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<Story asChild name="Splash screens">
	<StateGallery columns={3} minItemWidth="8rem">
		{#each splashes as splash (splash.label)}
			<StateGalleryItem label={splash.label} note={splash.note}>
				<div
					style:aspect-ratio="{splash.width} / {splash.height}"
					style:background-color={splash.background}
					class="splash flex h-60 items-center justify-center rounded-control ring-1 ring-line-default"
				>
					<div style:width="{splashLogo * 100}cqmin" class="mark aspect-square">
						{@html splash.mark}
					</div>
				</div>

				<p class="mt-2 text-xs text-fg-muted">{splash.files}</p>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<style>
	.mark :global(svg) {
		display: block;
		width: 100%;
		height: 100%;
	}

	.splash {
		container-type: size;
	}
</style>
