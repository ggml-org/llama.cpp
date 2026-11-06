<!--
	The brand's logo files as a grid of thumbnails, each with links to download it as SVG, WebP or PNG. The files are
	copied from ggml-org/llama.brand (logo/svg, logo/webp, logo/png, built by logo/scripts there); copy them again
	when the brand repo's logos change.
-->
<script lang="ts">
	import { Download } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

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

	/** A section of the page: one logo or platform, its assets in rows that each fit on one line. */
	interface Group {
		note: string;
		rows: { assets: Asset[]; label?: string }[];
		title: string;
	}

	const modeLabel = (mode: 'dark' | 'light') => (mode === 'light' ? 'Light' : 'Dark');

	const socialGroups: Group[] = social.map(({ banners, name, size, title }) => ({
		note: [
			`${size} × ${size} avatar`,
			...banners.map(([, label, w, h]) => `${label.toLowerCase()} ${w} × ${h}`)
		].join(' · '),
		rows: [
			{
				assets: (['light', 'dark'] as const).map((mode) => ({
					ground: 'cover' as const,
					label: modeLabel(mode),
					path: `social/${name}/${name}-avatar-${mode}`,
					round: true
				})),
				label: 'Avatar'
			},
			...banners.map(([kind, label]) => ({
				assets: ['llama', 'llama-cpp'].flatMap((lockup) =>
					(['light', 'dark'] as const).map((mode) => ({
						ground: 'cover' as const,
						label: `${lockup === 'llama' ? 'Llama' : 'llama.cpp'}, ${modeLabel(mode).toLowerCase()}`,
						path: `social/${name}/${name}-${kind}-${lockup}-${mode}`
					}))
				),
				label
			}))
		],
		title
	}));

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

	const groups: Group[] = $derived(section === 'logo' ? logoGroups : socialGroups);

	const formats: { format: Format; label: string }[] = [
		{ format: 'svg', label: 'SVG' },
		{ format: 'webp', label: 'WebP' },
		{ format: 'png', label: 'PNG' }
	];

	const fileName = (path: string, format: Format) => `${path.split('/').at(-1)}.${format}`;
</script>

<div class="mx-auto grid w-[min(72rem,calc(100%-2rem))] gap-20 py-10">
	<header class="grid gap-2">
		<h1 class="text-3xl font-bold tracking-tight">
			{section === 'logo' ? 'Logo assets' : 'Social profiles'}
		</h1>

		<p class="max-w-[64ch] text-sm text-muted-foreground">
			{section === 'logo'
				? 'The Llama icon and the Llama and llama.cpp lockups, each on its background and transparent'
				: 'Profile pictures and banners for X, LinkedIn and YouTube, at each platform’s size'}, to
			download as SVG, WebP or PNG. They come from
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

					<!-- every row on one line, in its group's columns so the rows line up; two to a line on phones -->
					<div
						style:--cols={Math.max(...group.rows.map((r) => r.assets.length))}
						class="grid grid-cols-2 items-start gap-4 sm:grid-cols-[repeat(var(--cols),minmax(0,1fr))]"
					>
						{#each row.assets as asset (asset.path)}
							<figure class="m-0 grid min-w-0 gap-2">
								<div
									style:aspect-ratio={ratio(asset.path)}
									style:padding={asset.transparent
										? `calc(12% / (${ratio(asset.path)}))`
										: undefined}
									class={[
										'min-w-0 overflow-hidden rounded-card border border-border/60',
										asset.round && 'mx-auto w-full max-w-40 rounded-full',
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
									<span class="truncate text-sm">{asset.label}</span>

									<span class="flex flex-wrap gap-1">
										{#each formats as f (f.format)}
											<Button
												class="h-6 px-1.5 font-mono text-xs text-muted-foreground has-[>svg]:px-1.5"
												download={fileName(asset.path, f.format)}
												href={url(asset.path, f.format)}
												size="sm"
												variant="ghost"><Download class="size-3" />{f.label}</Button
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
