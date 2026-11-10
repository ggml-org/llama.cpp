<!--
	llama.app's model page header, in the layouts proposed for it: how the maker's avatar, the name with its capability
	icons, the line of facts (maker, release) and the metadata chips sit together. Everything under the header is left
	out; the description stays, since it decides how the block reads.
-->
<script lang="ts">
	import type { Family } from './catalog';
	import { familyMem, modalitiesOf, reasons, releasedFor, sizesSummary, usesTools } from './catalog-helpers';
	import Description from './Description.svelte';
	import { ArrowLeft } from '@lucide/svelte';
	import { ModelCapabilityIcons, ModelMetadataItem, ModelOrgAvatar } from '$lib/components/models';

	let { family, layout }: { family: Family; layout: 'inline' | 'flush' | 'discover' | 'stacked' | 'eyebrow' | 'chips' } =
		$props();

	let mem = $derived(familyMem(family));
</script>

{#snippet back()}
	<a class="inline-flex w-fit items-center gap-1 text-body-sm text-muted-foreground hover:text-foreground" href="#back">
		<ArrowLeft aria-hidden="true" class="size-3.5" />
		Models
	</a>
{/snippet}

{#snippet icons()}
	<ModelCapabilityIcons
		gapClass="gap-2"
		iconSize="size-4"
		modalities={modalitiesOf(family.sizes)}
		supportsThinking={reasons(family.sizes)}
		supportsToolUse={usesTools(family.sizes)}
	/>
{/snippet}

{#snippet title(cls = 'text-heading @3xl:text-[1.75rem]')}
	<h1 class="flex flex-wrap items-center gap-x-3 gap-y-1 leading-tight tracking-[-0.015em] {cls}">
		{family.name}
		{@render icons()}
	</h1>
{/snippet}

{#snippet facts(cls = '')}
	<p class="text-body-sm text-muted-foreground tabular-nums {cls}">by {family.org} · Released {releasedFor(family)}</p>
{/snippet}

{#snippet chips(withFacts = false)}
	<div class="flex flex-wrap items-center gap-1.5">
		{#if withFacts}
			<ModelMetadataItem label="By" value={family.org} />
			<ModelMetadataItem label="Released" value={releasedFor(family)} />
		{/if}
		<ModelMetadataItem label="Parameters" value={sizesSummary(family)} />
		{#if mem}
			<ModelMetadataItem label="Memory" value="{mem} GB+" />
		{/if}
		<ModelMetadataItem label="License" value={family.license} />
	</div>
{/snippet}

{#snippet description()}
	<Description class="max-w-2xl text-body text-pretty text-muted-foreground" text={family.description} />
{/snippet}

<header class="flex flex-col gap-6 p-6 @3xl:p-12">
	{@render back()}

	{#if layout === 'inline'}
		<!-- the avatar on the name's line, the facts under the name -->
		<div class="grid grid-cols-[auto_minmax(0,1fr)] items-center gap-x-3 gap-y-1">
			<ModelOrgAvatar org={family.org} size="size-8 @3xl:size-9" />
			{@render title()}
			{@render facts('col-start-2')}
		</div>
	{:else if layout === 'flush'}
		<!-- the avatar on the name's line, the facts back at the avatar's edge -->
		<div class="flex flex-col gap-1">
			<div class="flex items-center gap-3">
				<ModelOrgAvatar org={family.org} size="size-8 @3xl:size-9" />
				{@render title()}
			</div>
			{@render facts()}
		</div>
	{:else if layout === 'discover'}
		<!-- Llama Chat's Discover details: a larger avatar beside the name and the facts together -->
		<div class="flex items-center gap-3">
			<ModelOrgAvatar org={family.org} size="size-12" />
			<div class="flex min-w-0 flex-col gap-0.5">
				{@render title('text-title @3xl:text-heading')}
				{@render facts()}
			</div>
		</div>
	{:else if layout === 'stacked'}
		<!-- the avatar on a line of its own above the name, like an app's icon over its title -->
		<div class="flex flex-col gap-3">
			<ModelOrgAvatar org={family.org} size="size-11" />
			<div class="flex flex-col gap-1">
				{@render title()}
				{@render facts()}
			</div>
		</div>
	{:else if layout === 'eyebrow'}
		<!-- the maker as an eyebrow over the name, with its avatar; the release with the chips -->
		<div class="flex flex-col gap-2">
			<p class="flex items-center gap-2 text-body-sm text-muted-foreground">
				<ModelOrgAvatar org={family.org} size="size-5" />
				{family.org} · {releasedFor(family)}
			</p>
			{@render title()}
		</div>
	{:else}
		<!-- no line of facts: the maker and the release join the chips -->
		<div class="flex items-center gap-3">
			<ModelOrgAvatar org={family.org} size="size-8 @3xl:size-9" />
			{@render title()}
		</div>
	{/if}

	{@render description()}
	{@render chips(layout === 'chips')}
</header>
