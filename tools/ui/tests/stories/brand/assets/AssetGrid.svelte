<!--
	The brand's logo files as a grid of thumbnails, each with links to download it as SVG, WebP or PNG. The files are
	copied from ggml-org/llama.brand (logo/svg, logo/webp, logo/png, built by logo/scripts there); copy them again
	when the brand repo's logos change.
-->
<script lang="ts">
	import { Download } from '@lucide/svelte';

	interface Props {
		/** the logo files (icon, lockups, covers) or the social profile pictures and banners */
		section: 'logo' | 'social';
	}

	let { section }: Props = $props();

	type Format = 'png' | 'svg' | 'webp';

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
		label: string;
		/** dark marks for light grounds (light) or light marks for dark ones (dark); social assets bring their own field */
		ground: 'cover' | 'dark' | 'light';
		path: string;
		/** a profile picture: shown in the circle the platforms crop it to */
		round?: boolean;
		/** a transparent file: shown over a checkerboard in its ground's shades */
		transparent?: boolean;
	}

	// social profiles: each platform's avatar and banners, light and dark, from logo/svg/social
	const social: {
		banners: [string, string, number, number][];
		name: string;
		size: number;
		title: string;
	}[] = [
		{ banners: [['header', 'Header', 1500, 500]], name: 'x', size: 400, title: 'X' },
		{
			banners: [
				['cover', 'Cover', 1584, 396],
				['page-cover', 'Page cover', 1128, 191]
			],
			name: 'linkedin',
			size: 400,
			title: 'LinkedIn'
		},
		{ banners: [['banner', 'Banner', 2560, 1440]], name: 'youtube', size: 800, title: 'YouTube' }
	];

	const socialGroups = social.map(({ banners, name, size, title }) => ({
		assets: [
			...(['light', 'dark'] as const).map((mode) => ({
				ground: 'cover' as const,
				label: `Avatar, ${mode}`,
				path: `social/${name}/${name}-avatar-${mode}`,
				round: true
			})),
			...banners.flatMap(([kind, label]) =>
				['llama', 'llama-cpp'].flatMap((lockup) =>
					(['light', 'dark'] as const).map((mode) => ({
						ground: 'cover' as const,
						label: `${label}, ${lockup === 'llama' ? 'Llama' : 'llama.cpp'}, ${mode}`,
						path: `social/${name}/${name}-${kind}-${lockup}-${mode}`
					}))
				)
			)
		],
		note: [
			`${size} × ${size} avatar`,
			...banners.map(([, label, w, h]) => `${label.toLowerCase()} ${w} × ${h}`)
		].join(' · '),
		title
	}));

	/**
	 * Each icon and lockup in both of its versions, for light and for dark: on its background (-bg, shown as it is) and
	 * transparent (shown over a checkerboard in the ground it's drawn for).
	 */
	function versions(name: string, path: (mode: string) => string): Asset[] {
		return (['light', 'dark'] as const).flatMap((mode) => [
			{
				ground: 'cover' as const,
				label: `${name}${mode === 'light' ? 'Light' : 'Dark'}, on background`,
				path: `${path(mode)}-bg`
			},
			{
				ground: mode,
				label: `${name}${mode === 'light' ? 'Light' : 'Dark'}, transparent`,
				path: path(mode),
				transparent: true
			}
		]);
	}

	const logoGroups: { assets: Asset[]; note: string; title: string }[] = [
		{
			assets: versions('', (m) => `icon/icon-${m}`),
			note: '600 × 600 · WebP and PNG at 1200 × 1200 · on background, optically centred',
			title: 'Icon'
		},
		{
			assets: [
				...versions('Llama · ', (m) => `lockup/llama/logo-llama-${m}`),
				...versions('llama.cpp · ', (m) => `lockup/llama-cpp/logo-llama-cpp-${m}`)
			],
			note: 'The icon with the wordmark: Llama in Llama Sans Bold at its Display size, llama.cpp in Llama Mono Bold · WebP and PNG at 2×',
			title: 'Lockups'
		}
	];

	const groups: { assets: Asset[]; note: string; title: string }[] = $derived(
		section === 'logo' ? logoGroups : socialGroups
	);

	const formats: { format: Format; label: string }[] = [
		{ format: 'svg', label: 'SVG' },
		{ format: 'webp', label: 'WebP' },
		{ format: 'png', label: 'PNG' }
	];

	const fileName = (path: string, format: Format) => `${path.split('/').at(-1)}.${format}`;
</script>

<div class="mx-auto grid w-[min(72rem,calc(100%-2rem))] gap-12 py-10">
	<header class="grid gap-2">
		<h1 class="text-3xl font-bold tracking-tight">
			{section === 'logo' ? 'Logo assets' : 'Social profiles'}
		</h1>

		<p class="max-w-[64ch] text-sm text-muted-foreground">
			{section === 'logo'
				? 'The Llama icon and the Llama and llama.cpp lockups, each on its background and transparent,'
				: 'Profile pictures and banners for X, LinkedIn and YouTube, at each platform’s size'}, to
			download as SVG, WebP or PNG. They come from
			<a class="underline underline-offset-2" href="https://github.com/ggml-org/llama.brand"
				>ggml-org/llama.brand</a
			>, under CC BY-NC 4.0 with the permissions in its BRAND-USAGE.md: use them as they are to
			refer to Llama.app and Llama.cpp, without changing the mark.
		</p>
	</header>

	{#each groups as group (group.title)}
		<section class="grid gap-4">
			<div class="flex items-baseline gap-3">
				<h2 class="text-lg font-semibold">{group.title}</h2>

				<p class="text-xs text-muted-foreground">{group.note}</p>
			</div>

			<div
				class={[
					'grid items-end gap-4',
					group.title === 'Icon' || section === 'social'
						? 'grid-cols-[repeat(auto-fill,minmax(14rem,1fr))]'
						: 'grid-cols-[repeat(auto-fill,minmax(22rem,1fr))]'
				]}
			>
				{#each group.assets as asset (asset.path)}
					<figure class="m-0 grid min-w-0 gap-2">
						<div
							style:aspect-ratio={ratio(asset.path)}
							class={[
								'min-w-0 overflow-hidden rounded-card border border-border/60',

								asset.round && 'mx-auto w-2/3 rounded-full',
								asset.ground === 'light' && 'bg-[#fafafa]',
								asset.ground === 'dark' && 'bg-[#111111]',
								asset.transparent && (asset.ground === 'light' ? 'checker-light' : 'checker-dark')
							]}
						>
							<img
								alt={asset.label}
								class={[
									'block size-full',
									asset.ground === 'cover' ? 'object-cover' : 'object-contain'
								]}
								src={url(asset.path, 'svg')}
							/>
						</div>

						<figcaption class="flex flex-wrap items-center justify-between gap-x-2 gap-y-1">
							<span class="truncate text-sm">{asset.label}</span>

							<span class="flex shrink-0 gap-1">
								{#each formats as f (f.format)}
									<a
										class="inline-flex items-center gap-1 rounded-control px-1.5 py-0.5 font-mono text-xs text-muted-foreground hover:bg-surface-muted hover:text-foreground"
										download={fileName(asset.path, f.format)}
										href={url(asset.path, f.format)}><Download class="size-3" />{f.label}</a
									>
								{/each}
							</span>
						</figcaption>
					</figure>
				{/each}
			</div>
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
