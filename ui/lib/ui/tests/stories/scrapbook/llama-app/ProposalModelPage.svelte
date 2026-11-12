<!--
	A family's page on llama.app, proposed after Llama Chat's Discover details (ModelsDiscoverDetails): the header with
	the maker's avatar, the name and its capability icons, and the link to Hugging Face; a line of facts and the
	metadata chips; then each size in Discover's quant panel, one row per bit depth, every chip an install link and the
	row's label opening its command.
-->
<script lang="ts">
	import type { Build, Family, Size } from './catalog';
	import {
		bitDepthOf,
		deeplink,
		familyMem,
		memFor,
		modalitiesOf,
		reasons,
		releasedFor,
		paramsOf,
		usesTools
	} from './catalog-helpers';
	import Description from './Description.svelte';
	import { CalendarDays, ExternalLink, Layers, MemoryStick } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import {
		ModelCapabilityIcons,
		ModelId,
		ModelMetadataItem,
		ModelOrgAvatar,
		ModelQuantChip,
		ModelQuantOptions,
		ModelQuantRow
	} from '$lib/components/models';

	let { family }: { family: Family } = $props();

	let mem = $derived(familyMem(family));
	let params = $derived([...new Set(family.sizes.map(paramsOf))]);
	let repos = $derived([...new Set(family.sizes.flatMap((size) => size.builds.map((build) => build.repo)))]);

	/** A size's builds in bit-depth rows, highest precision first. */
	function rowsOf(size: Size): { bitDepth: number; builds: Build[] }[] {
		const rows = new Map<number, Build[]>();

		for (const build of size.builds) {
			const depth = bitDepthOf(build.quant);

			rows.set(depth, [...(rows.get(depth) ?? []), build]);
		}

		return [...rows].sort(([a], [b]) => b - a).map(([bitDepth, builds]) => ({ bitDepth, builds }));
	}

	const command = (build: Build) => `llama serve -hf ${build.repo}${build.quant ? `:${build.quant}` : ''}`;
</script>

<main class="flex flex-col gap-8 px-5 py-8 @3xl:px-10 @3xl:py-12">
	<!-- the details header: avatar, name with its capability icons, the Hub link at the right -->
	<header class="flex flex-col gap-4">
		<div class="flex flex-wrap items-start justify-between gap-4">
			<div class="flex min-w-0 items-center gap-3">
				<ModelOrgAvatar
					org={family.org}
					quantOrg="ggml-org"
					quantPositionClass="-bottom-1.5 -right-1.5"
					quantSize="size-6"
					size="size-12"
				/>

				<div class="flex min-w-0 flex-col gap-1">
					<h1 class="flex flex-wrap items-center gap-x-3 gap-y-1 text-[1.75rem] leading-tight font-bold tracking-[-0.02em] @3xl:text-[2.25rem]">
						{family.name}

						<ModelCapabilityIcons
							gapClass="gap-2"
							iconSize="size-icon-md"
							modalities={modalitiesOf(family.sizes)}
							supportsThinking={reasons(family.sizes)}
							supportsToolUse={usesTools(family.sizes)}
						/>
					</h1>

					<!-- Discover's base-model line: who made it, who made the GGUF -->
					<span class="truncate text-xs text-muted-foreground">{family.org} · GGUF by ggml-org</span>
				</div>
			</div>

			<Button href="https://huggingface.co/{repos[0]}" size="sm" variant="secondary">
				View on Hugging Face
				<ExternalLink />
			</Button>
		</div>

		<!-- the Hub stats line, with what the catalog knows: released, sizes, the memory the smallest needs -->
		<div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-muted-foreground tabular-nums">
			<span class="inline-flex items-center gap-1.5"><CalendarDays class="size-icon-sm" />{releasedFor(family)}</span>
			<span class="inline-flex items-center gap-1.5"><Layers class="size-icon-sm" />{family.sizes.length} {family.sizes.length === 1 ? 'size' : 'sizes'}</span>

			{#if mem}
				<span class="inline-flex items-center gap-1.5"><MemoryStick class="size-icon-sm" />{mem} GB+ memory</span>
			{/if}
		</div>

		<Description class="max-w-2xl text-body text-muted-foreground" text={family.description} />

		<div class="flex flex-wrap items-center gap-1.5">
			<!-- the span only: the stats line above counts the sizes, and each has its own section below -->
			<ModelMetadataItem
				label="Parameters"
				value={params.length === 1 ? params[0] : `${params[0]}–${params[params.length - 1]}`}
			/>
			<ModelMetadataItem label="License" value={family.license} />

			{#if family.nonCommercial}
				<Badge variant="warning">Non-commercial</Badge>
			{/if}
		</div>
	</header>

	{#each family.sizes as size (size.name)}
		<section class="flex flex-col gap-3">
			<ModelId
				class="text-title"
				modalities={modalitiesOf([size])}
				name={size.name}
				supportsThinking={reasons([size])}
				supportsToolUse={usesTools([size])}
			/>

			<ModelQuantOptions>
				{#each rowsOf(size) as row (row.bitDepth)}
					<ModelQuantRow bitDepth={row.bitDepth} memGb={memFor(row.builds[0])}>
						{#each row.builds as build (build.repo + build.quant)}
							<ModelQuantChip
								actionLabel="Install {build.quant?.toUpperCase()} in Llama"
								href={deeplink(build)}
								label={build.quant?.toUpperCase() ?? 'GGUF'}
								sizeBytes={build.sizeBytes}
							/>
						{/each}

						{#snippet command()}
							<pre
								class="overflow-x-auto rounded-panel bg-(--surface-code) px-3 py-2 text-xs text-(--fg-code)">{command(
									row.builds[0]
								)}</pre>
						{/snippet}
					</ModelQuantRow>
				{/each}
			</ModelQuantOptions>
		</section>
	{/each}

	<p class="max-w-2xl text-body-sm text-muted-foreground">
		A chip installs that quant in Llama on your Mac. A row's label shows the command to run it with llama.cpp
		instead.
	</p>
</main>
