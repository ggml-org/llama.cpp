<!--
	The brand's logo files as a grid of thumbnails, each with links to download it as SVG, WebP or PNG. The files are
	copied from ggml-org/llama.brand (logo/svg, logo/webp, logo/png, built by logo/scripts there); copy them again
	when the brand repo's logos change.
-->
<script lang="ts">
	import { Download } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	interface Props {
		/** the logo files (icon, lockups), the social profile pictures and banners, or the app icons */
		section: 'app' | 'logo' | 'social';
	}

	let { section }: Props = $props();

	type Format = 'png' | 'svg' | 'webp';

	const formatLabel: Record<Format, string> = { png: 'PNG', svg: 'SVG', webp: 'WebP' };

	const files = import.meta.glob('./files/logo/**/*.{svg,webp,png}', {
		eager: true,
		import: 'default',
		query: '?url'
	}) as Record<string, string>;

	const svgs = import.meta.glob('./files/logo/svg/**/*.svg', {
		eager: true,
		import: 'default',
		query: '?raw'
	}) as Record<string, string>;

	/** The URL of one asset, by its path under logo/<format>/ without the extension. */
	function url(path: string, format: Format): string {
		return files[`./files/logo/${format}/${path}.${format}`];
	}

	/** An asset's own proportions, from its SVG's viewBox, as an aspect-ratio. */
	function ratio(path: string): string {
		const [, , w, h] = svgs[`./files/logo/svg/${path}.svg`]
			.match(/viewBox="([^"]+)"/)![1]
			.split(/\s+/);

		return `${w} / ${h}`;
	}

	interface Asset {
		/** the files to download, when not this asset's own SVG, WebP and PNG */
		downloads?: { href: string; label: string; name: string }[];
		label: string;
		/** dark marks for light grounds (light) or light marks for dark ones (dark); social assets bring their own field */
		ground: 'cover' | 'dark' | 'light';
		path: string;
		/** a profile picture: shown in the circle the platforms crop it to */
		round?: boolean;
		/** a transparent file: shown over a checkerboard in its ground's shades */
		transparent?: boolean;
		/** a transparent file with its own margin (an app icon's canvas): shown as it is, with none added */
		unpadded?: boolean;
	}

	// social profiles: each platform's banners, light and dark, from logo/svg/social (the avatars are one section)
	const social: {
		banners: [string, string, number, number][];
		name: string;
		title: string;
	}[] = [
		{ banners: [['header', 'Header', 1500, 500]], name: 'x', title: 'X' },
		{
			banners: [
				['cover', 'Cover', 1584, 396],
				['page-cover', 'Page cover', 1128, 191]
			],
			name: 'linkedin',
			title: 'LinkedIn'
		},
		{ banners: [['banner', 'Banner', 2560, 1440]], name: 'youtube', title: 'YouTube' }
	];

	/** A section of the page: one logo or platform, its assets in rows that each fit on one line. */
	interface Group {
		note: string;
		rows: { assets: Asset[]; label?: string }[];
		title: string;
	}

	const modeLabel = (mode: 'dark' | 'light') => (mode === 'light' ? 'Light' : 'Dark');

	/**
	 * One avatar for every platform, light and dark: X's and LinkedIn's are the same 400 × 400 picture, YouTube's the
	 * same at 800 × 800, so each offers its SVG and its WebP and PNG at both sizes.
	 */
	const avatarGroup: Group = {
		note: 'The icon in a square, for every platform’s profile picture, which crops it to a circle · WebP and PNG at 400 × 400 (X, LinkedIn) and 800 × 800 (YouTube)',
		rows: [
			{
				assets: (['light', 'dark'] as const).map((mode) => ({
					downloads: [
						{ format: 'svg' as const, name: 'x', size: '' },
						...(['webp', 'png'] as const).flatMap((format) => [
							{ format, name: 'x', size: '400' },
							{ format, name: 'youtube', size: '800' }
						])
					].map(({ format, name, size }) => ({
						href: url(`social/${name}/${name}-avatar-${mode}`, format),
						label: `${formatLabel[format]}${size && ` ${size}`}`,
						name: `avatar-${mode}${size && `-${size}`}.${format}`
					})),
					ground: 'cover' as const,
					label: modeLabel(mode),
					path: `social/x/x-avatar-${mode}`,
					round: true
				}))
			}
		],
		title: 'Avatar'
	};

	const socialGroups: Group[] = [
		avatarGroup,
		...social.map(({ banners, name, title }) => ({
			note: banners.map(([, label, w, h]) => `${label} ${w} × ${h}`).join(' · '),
			rows: [
				...banners.map(([kind, label]) => ({
					assets: ['llama', 'llama-cpp'].flatMap((lockup) =>
						(['light', 'dark'] as const).map((mode) => ({
							ground: 'cover' as const,
							label: `${lockup === 'llama' ? 'Llama' : 'llama.cpp'}, ${modeLabel(mode).toLowerCase()}`,
							path: `social/${name}/${name}-${kind}-${lockup}-${mode}`
						}))
					),
					label: banners.length > 1 ? label : undefined
				}))
			],
			title
		}))
	];

	/**
	 * Each icon and lockup in both of its versions, for light and for dark: on its background (-bg, shown as it is) and
	 * transparent (shown over a checkerboard in the ground it's drawn for).
	 */
	function versions(path: (mode: string) => string): Asset[] {
		return (['light', 'dark'] as const).flatMap((mode) => [
			{
				ground: 'cover' as const,
				label: `${modeLabel(mode)}, on background`,
				path: `${path(mode)}-bg`
			},
			{
				ground: mode,
				label: `${modeLabel(mode)}, transparent`,
				path: path(mode),
				transparent: true
			}
		]);
	}

	const logoGroups: Group[] = [
		{
			note: 'The L-shaped mark alone · SVG 600 × 600, WebP and PNG at 1200 × 1200 · on its background, optically centred',
			rows: [{ assets: versions((m) => `icon/icon-${m}`) }],
			title: 'Icon'
		},
		{
			note: 'The icon with Llama in Llama Sans Bold · SVG 2342 × 600, WebP and PNG at 2×',
			rows: [{ assets: versions((m) => `lockup/llama/logo-llama-${m}`) }],
			title: 'Llama lockup'
		},
		{
			note: 'The icon with llama.cpp in Llama Mono Bold, .cpp in cinder · SVG 3926 × 600, WebP and PNG at 2× · on neutral grounds only',
			rows: [{ assets: versions((m) => `lockup/llama-cpp/logo-llama-cpp-${m}`) }],
			title: 'llama.cpp lockup'
		}
	];

	/**
	 * An app icon in both modes, from logo/svg/app/<platform>: on its field (shown as it is) or, transparent, over a
	 * checkerboard in its mode's shades.
	 */
	function app(platform: string, name: string, label: string, transparent = false): Asset[] {
		return (['light', 'dark'] as const).map((mode) => ({
			ground: transparent ? mode : ('cover' as const),
			label: `${label}, ${modeLabel(mode).toLowerCase()}`,
			path: `app/${platform}/${name}-${mode}`,
			transparent,
			unpadded: true
		}));
	}

	// the icon for each platform, from logo/scripts/build_app_icons.py: full-bleed squares carry it at 67.2%, inside
	// the 80% safe circle, and every one is lifted 1/150 of its height
	const appGroups: Group[] = [
		{
			note: 'For the web app manifest · the round icon (any) at 512 × 512, the icon 57.1% of the circle, transparent outside it · the maskable icon at 512 × 512, full bleed, the icon inside the 80% safe circle',
			rows: [
				{
					assets: [
						...app('pwa', 'pwa-512x512', 'Round', true),
						...app('pwa', 'pwa-maskable-512x512', 'Maskable')
					]
				}
			],
			title: 'PWA'
		},
		{
			note: 'Full-bleed squares that iOS rounds: the touch icon for the web at 180 × 180 and the iOS app icon at 1024 × 1024 · the macOS app icon at 1024 × 1024, on Apple’s 824 × 824 rounded square, transparent around it',
			rows: [
				{
					assets: [
						...app('apple', 'apple-touch-icon-180x180', 'Touch icon'),
						...app('apple', 'ios-1024x1024', 'iOS'),
						...app('apple', 'macos-1024x1024', 'macOS', true)
					]
				}
			],
			title: 'Apple'
		},
		{
			note: 'The adaptive icon’s two layers at 432 × 432 (108 dp at 4×), the icon inside the 66 dp circle every launcher shows · the Play Store icon at 512 × 512 · one monochrome layer, which the launcher tints for themed icons',
			rows: [
				{
					assets: [
						...app('android', 'android-adaptive-foreground-432x432', 'Foreground', true),
						...app('android', 'android-adaptive-background-432x432', 'Background'),
						...app('android', 'android-play-512x512', 'Play Store'),
						{
							ground: 'light',
							label: 'Monochrome',
							path: 'app/android/android-monochrome-432x432',
							transparent: true,
							unpadded: true
						}
					]
				}
			],
			title: 'Android'
		},
		{
			note: 'The app icon at 256 × 256, transparent with a 1/16 margin · the Start and Store tiles at 150 × 150 and 310 × 150, the icon at half their height',
			rows: [
				{
					assets: [
						...app('windows', 'windows-app-icon-256x256', 'App icon', true),
						...app('windows', 'windows-tile-150x150', 'Tile'),
						...app('windows', 'windows-tile-wide-310x150', 'Wide tile')
					]
				}
			],
			title: 'Windows'
		}
	];

	const groups: Group[] = $derived(
		section === 'logo' ? logoGroups : section === 'social' ? socialGroups : appGroups
	);

	// the page's columns: as many as its longest row, so every thumbnail is the same width
	const columns = $derived(Math.max(...groups.flatMap((g) => g.rows.map((r) => r.assets.length))));

	/** An asset's downloads: its own SVG, WebP and PNG, unless it names others. */
	const downloads = (asset: Asset) =>
		asset.downloads ??
		(['svg', 'webp', 'png'] as const).map((format) => ({
			href: url(asset.path, format),
			label: formatLabel[format],
			name: `${asset.path.split('/').at(-1)}.${format}`
		}));
</script>

<div class="mx-auto grid w-[min(72rem,calc(100%-2rem))] gap-20 py-10">
	<header class="grid gap-2">
		<h1 class="text-3xl font-bold tracking-tight">
			{section === 'logo' ? 'Logo assets' : section === 'social' ? 'Social profiles' : 'App icons'}
		</h1>

		<p class="max-w-[64ch] text-sm text-muted-foreground">
			{section === 'logo'
				? 'The Llama icon and the Llama and llama.cpp lockups, each on its background and transparent'
				: section === 'social'
					? 'Profile pictures and banners for X, LinkedIn and YouTube, at each platform’s size'
					: 'The icon for a PWA, Apple’s platforms, Android and Windows, light and dark, at each platform’s size and in its shape'},
			to download as SVG, WebP or PNG. They come from
			<a class="underline underline-offset-2" href="https://github.com/ggml-org/llama.brand"
				>ggml-org/llama.brand</a
			>, under CC BY-NC 4.0 with the permissions in its BRAND-USAGE.md: use them as they are to
			refer to Llama.app and Llama.cpp, without changing the mark.
		</p>
	</header>

	{#each groups as group (group.title)}
		<section class="grid gap-6">
			<div class="grid gap-1">
				<h2 class="text-lg font-semibold">{group.title}</h2>

				<p class="text-xs text-muted-foreground">{group.note}</p>
			</div>

			{#each group.rows as row, i (row.label ?? i)}
				<div class="grid gap-3">
					{#if row.label}
						<h3 class="text-sm font-medium text-muted-foreground">{row.label}</h3>
					{/if}

					<!-- every row on one line, in the page's columns so the rows and sections line up; two to a line on phones -->
					<div
						style:--cols={columns}
						class="grid grid-cols-2 items-start gap-4 sm:grid-cols-[repeat(var(--cols),minmax(0,1fr))]"
					>
						{#each row.assets as asset (asset.path)}
							<figure class="m-0 grid min-w-0 gap-2">
								<div
									style:aspect-ratio={ratio(asset.path)}
									style:padding={asset.transparent && !asset.unpadded
										? `calc(12% / (${ratio(asset.path)}))`
										: undefined}
									class={[
										'min-w-0 overflow-hidden rounded-card border border-border/60',
										asset.round && 'mx-auto w-full max-w-48 rounded-full',
										asset.ground === 'light' && 'bg-[#fafafa]',
										asset.ground === 'dark' && 'bg-[#111111]',
										asset.transparent &&
											(asset.ground === 'light' ? 'checker-light' : 'checker-dark')
									]}
								>
									<img
										alt={`${group.title}${row.label ? ` ${row.label.toLowerCase()}` : ''}, ${asset.label}`}
										class={[
											'block size-full',
											asset.transparent ? 'object-contain' : 'object-cover'
										]}
										src={url(asset.path, 'svg')}
									/>
								</div>

								<figcaption class={['grid gap-1', asset.round && 'justify-items-center']}>
									<span class="text-sm">{asset.label}</span>

									<span class={['flex flex-wrap gap-1', asset.round && 'justify-center']}>
										<!-- the click stops here: @storybook/sveltekit's MockProvider cancels every link click that
										     reaches the window, to log it as navigation, which would swallow the download -->
										{#each downloads(asset) as file (file.name)}
											<Button
												class="h-6 px-1.5 font-mono text-xs text-muted-foreground has-[>svg]:px-1.5"
												download={file.name}
												href={file.href}
												onclick={(event: MouseEvent) => event.stopPropagation()}
												size="sm"
												variant="ghost"><Download class="size-3" />{file.label}</Button
											>
										{/each}
									</span>
								</figcaption>
							</figure>
						{/each}
					</div>
				</div>
			{/each}
		</section>
	{/each}
</div>

<style>
	/* the checkerboard that marks a transparent file, in its ground's two shades */
	.checker-light,
	.checker-dark {
		background-image: repeating-conic-gradient(var(--checker) 0 25%, transparent 0 50%);
		background-size: 16px 16px;
	}
	.checker-light {
		--checker: #ebebeb;
	}
	.checker-dark {
		--checker: #1f1f1f;
	}
</style>
