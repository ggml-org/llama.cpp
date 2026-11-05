<!--
	The brand's logo files as a grid of thumbnails, each with links to download it as SVG, WebP or PNG. The files are
	copied from ggml-org/llama.brand (logo/svg, logo/webp, logo/png, built by logo/scripts there); copy them again
	when the brand repo's logos change.
-->
<script lang="ts">
	import { Download } from '@lucide/svelte';

	type Format = 'png' | 'svg' | 'webp';

	const files = import.meta.glob('./assets/logo/**/*.{svg,webp,png}', {
		eager: true,
		import: 'default',
		query: '?url'
	}) as Record<string, string>;

	/** The URL of one asset, by its path under logo/<format>/ without the extension. */
	function url(path: string, format: Format): string {
		return files[`./assets/logo/${format}/${path}.${format}`];
	}

	interface Asset {
		label: string;
		/** dark marks for light grounds (light) or light marks for dark ones (dark); covers bring their own field */
		ground: 'cover' | 'dark' | 'light';
		path: string;
	}

	const groups: { assets: Asset[]; note: string; title: string }[] = [
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
			note: 'The icon with the wordmark in Llama Sans Display · WebP and PNG at 2×',
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

	const formats: { format: Format; label: string }[] = [
		{ format: 'svg', label: 'SVG' },
		{ format: 'webp', label: 'WebP' },
		{ format: 'png', label: 'PNG' }
	];

	const fileName = (path: string, format: Format) => `${path.split('/').at(-1)}.${format}`;
</script>

<div class="mx-auto grid w-[min(72rem,calc(100%-2rem))] gap-12 py-10">
	<header class="grid gap-2">
		<h1 class="text-3xl font-bold tracking-tight">Assets</h1>

		<p class="max-w-[64ch] text-sm text-muted-foreground">
			The Llama icon, the Llama and llama.cpp lockups and the covers, to download as SVG, WebP or
			PNG. They come from
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
					'grid gap-4',
					group.title === 'Icon'
						? 'grid-cols-[repeat(auto-fill,minmax(14rem,1fr))]'
						: 'grid-cols-[repeat(auto-fill,minmax(22rem,1fr))]'
				]}
			>
				{#each group.assets as asset (asset.path)}
					<figure class="m-0 grid min-w-0 gap-2">
						<div
							class={[
								'grid min-w-0 place-items-center overflow-hidden rounded-card border border-border/60',
								asset.ground === 'cover' ? 'aspect-[3/1]' : 'aspect-[4/3] p-8',
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
