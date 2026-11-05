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

	/** The URL of one asset, by its path under logo/<format>/ without the extension. */
	function url(path: string, format: Format): string {
		return files[`./files/logo/${format}/${path}.${format}`];
	}

	interface Asset {
		label: string;
		/** dark marks for light grounds (light) or light marks for dark ones (dark); covers bring their own field */
		ground: 'cover' | 'dark' | 'light';
		path: string;
		/** the asset's own proportions, for covers and social banners (default 3 / 1) */
		ratio?: string;
		/** a profile picture: shown in the circle the platforms crop it to */
		round?: boolean;
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
				ratio: '1 / 1',
				round: true
			})),
			...banners.flatMap(([kind, label, w, h]) =>
				['llama', 'llama-cpp'].flatMap((lockup) =>
					(['light', 'dark'] as const).map((mode) => ({
						ground: 'cover' as const,
						label: `${label}, ${lockup === 'llama' ? 'Llama' : 'llama.cpp'}, ${mode}`,
						path: `social/${name}/${name}-${kind}-${lockup}-${mode}`,
						ratio: `${w} / ${h}`
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

	const logoGroups: { assets: Asset[]; note: string; title: string }[] = [
		{
			assets: [
				{ ground: 'light', label: 'On light', path: 'icon/icon-light' },
				{ ground: 'dark', label: 'On dark', path: 'icon/icon-dark' }
			],
			note: '600 × 600 · WebP and PNG at 1200 × 1200',
			title: 'Icon'
		},
		{
			assets: [
				{ ground: 'light', label: 'Llama, on light', path: 'lockup/llama/logo-llama-light' },
				{ ground: 'dark', label: 'Llama, on dark', path: 'lockup/llama/logo-llama-dark' },
				{
					ground: 'light',
					label: 'llama.cpp, on light',
					path: 'lockup/llama-cpp/logo-llama-cpp-light'
				},
				{
					ground: 'dark',
					label: 'llama.cpp, on dark',
					path: 'lockup/llama-cpp/logo-llama-cpp-dark'
				}
			],
			note: 'The icon with the wordmark in Llama Sans Bold at its Display size · WebP and PNG at 2×',
			title: 'Lockups'
		},
		{
			assets: [
				{ ground: 'cover', label: 'Llama, light', path: 'cover/llama/cover-llama-light' },
				{ ground: 'cover', label: 'Llama, dark', path: 'cover/llama/cover-llama-dark' },
				{
					ground: 'cover',
					label: 'llama.cpp, light',
					path: 'cover/llama-cpp/cover-llama-cpp-light'
				},
				{ ground: 'cover', label: 'llama.cpp, dark', path: 'cover/llama-cpp/cover-llama-cpp-dark' }
			],
			note: '7500 × 2500, for banners and social cards',
			title: 'Covers'
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
				? 'The Llama icon, the Llama and llama.cpp lockups and the covers'
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
							style:aspect-ratio={asset.ground === 'cover' ? (asset.ratio ?? '3 / 1') : undefined}
							class={[
								'grid min-w-0 place-items-center overflow-hidden rounded-card border border-border/60',
								asset.ground === 'cover' ? '' : 'aspect-[4/3] p-8',
								asset.round && 'mx-auto w-2/3 rounded-full',
								asset.ground === 'light' && 'bg-[#fafafa]',
								asset.ground === 'dark' && 'bg-[#111111]'
							]}
						>
							<img
								alt={asset.label}
								class={asset.ground === 'cover'
									? 'size-full object-cover'
									: 'h-auto max-h-full w-full max-w-full object-contain'}
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
