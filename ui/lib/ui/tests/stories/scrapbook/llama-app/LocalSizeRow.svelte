<!--
	One size on llama.app's model page (avatar, ModelId, the Hugging Face link, then a quant row with each build's
	Install button), with the ways proposed to say which of its builds are already in Llama on this computer:
	- status: a lamp and a word under the name, as Llama Chat's Models Manager shows a model's state
	- badge: a status Badge beside the name's badges
	- build: on the build itself, its Install button giving way to the state and a Chat button
	- both: the status line under the name and the state on the build
-->
<script lang="ts">
	import type { Build, Size } from './catalog';
	import { bitDepthOf, memFor, modalitiesOf, reasons, usesTools } from './catalog-helpers';
	import { ArrowUpRight, Check, Download, MessageCircle } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { ModelId, ModelOrgAvatar, ModelQuantOptions, ModelQuantRow } from '$lib/components/models';
	import { formatFileSize } from '$lib/models/format';

	let {
		org,
		size,
		state,
		variant
	}: {
		org: string;
		size: Size;
		state: 'downloaded' | 'loaded';
		variant: 'status' | 'badge' | 'build' | 'both';
	} = $props();

	// the first build is the one on this computer
	let local = $derived(size.builds[0]);
	let label = $derived(state === 'loaded' ? 'Loaded' : 'Downloaded');
	let lamp = $derived(state === 'loaded' ? 'status-lamp orb-success' : 'bg-muted-foreground/60');
	let params = $derived(size.params?.match(/^E?[\d.]+[BM](?:-A[\d.]+B)?/)?.[0]);
	let name = $derived(params ? size.name.replace(size.params ?? '', '').trim() : size.name);
	let rows = $derived(
		[...new Set(size.builds.map((b) => bitDepthOf(b.quant)))]
			.sort((a, b) => b - a)
			.map((depth) => ({ builds: size.builds.filter((b) => bitDepthOf(b.quant) === depth), depth }))
	);

	const isLocal = (build: Build) => build === local;
</script>

<section class="flex flex-col gap-3">
	<div class="flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
		<div class="flex min-w-0 flex-col gap-1">
			<h2 class="flex min-w-0 items-center gap-2 text-body leading-snug">
				<ModelOrgAvatar {org} size="size-5" />
				<ModelId
					modalities={modalitiesOf([size])}
					{name}
					{params}
					supportsThinking={reasons([size])}
					supportsToolUse={usesTools([size])}
					wrap
				/>
				{#if variant === 'badge'}
					<Badge variant={state === 'loaded' ? 'success' : 'secondary'}>{label} · {local.quant?.toUpperCase()}</Badge>
				{/if}
			</h2>

			{#if variant === 'status' || variant === 'both'}
				<!-- Models Manager's status line: the lamp, the state, which build -->
				<p class="flex items-center gap-1.5 pl-7 text-caption text-muted-foreground">
					<span class="size-2 shrink-0 rounded-full {lamp}"></span>
					{label} in Llama{#if variant === 'status'}&nbsp;· {local.quant?.toUpperCase()}{/if}
				</p>
			{/if}
		</div>

		<div class="flex items-center gap-2">
			<Button size="sm" variant="ghost">Hugging Face <ArrowUpRight /></Button>
			{#if variant !== 'build' && variant !== 'both'}
				<Button size="sm"><MessageCircle />Chat with model</Button>
			{/if}
		</div>
	</div>

	<ModelQuantOptions>
		{#each rows as row (row.depth)}
			<ModelQuantRow bitDepth={row.depth} memGb={memFor(row.builds[0])}>
				{#each row.builds as build (build.quant)}
					<span class="inline-flex items-center gap-3">
						<span class="flex items-baseline gap-2 tabular-nums">
							<span class="font-mono text-xs font-medium">{build.quant?.toUpperCase()}</span>
							<span class="text-xs text-muted-foreground">{formatFileSize(build.sizeBytes ?? 0)}</span>
						</span>

						{#if (variant === 'build' || variant === 'both') && isLocal(build)}
							<span class="inline-flex items-center gap-1 text-caption {state === 'loaded' ? 'text-fg-success' : 'text-muted-foreground'}">
								<Check class="size-icon-xs" />{label}
							</span>
							<Button size="sm"><MessageCircle />Chat</Button>
						{:else if variant !== 'build' && variant !== 'both' && isLocal(build)}
							<Button size="sm" variant="secondary"><Check />Installed</Button>
						{:else}
							<Button size="sm"><Download />Install</Button>
						{/if}
					</span>
				{/each}
			</ModelQuantRow>
		{/each}
	</ModelQuantOptions>
</section>
