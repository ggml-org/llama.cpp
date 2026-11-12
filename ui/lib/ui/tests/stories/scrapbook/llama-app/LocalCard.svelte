<!--
	llama.app's catalog card (ModelFamilyCard's markup, on ModelCard) with the ways proposed to say the family is already
	in Llama on this computer: `downloaded`, or `loaded` (in memory, ready to chat).
	- badge: a status Badge at the foot's end, where the release month was
	- avatar: a status lamp on the avatar's corner, like an app's presence dot
	- line: a lamp and a word in the sizes line under the name
	- corner: a check at the top right, before the capability icons
-->
<script lang="ts">
	import type { Family } from './catalog';
	import { familyMem, modalitiesOf, reasons, releasedFor, sizeRange, sizesSummary, usesTools } from './catalog-helpers';
	import { CircleCheck, Database, MemoryStick } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { ModelCapabilityIcons, ModelCard, ModelOrgAvatar } from '$lib/components/models';
	import { formatSizeRange } from '$lib/models/format';

	let {
		family,
		state,
		variant
	}: { family: Family; state: 'downloaded' | 'loaded'; variant: 'badge' | 'avatar' | 'line' | 'corner' } = $props();

	let range = $derived(sizeRange(family.sizes));
	let mem = $derived(familyMem(family));
	let label = $derived(state === 'loaded' ? 'Loaded' : 'Downloaded');
	// loaded is live, the success lamp as in Llama Chat's Models Manager; downloaded is at rest, the quiet one
	let lamp = $derived(state === 'loaded' ? 'status-lamp orb-success' : 'bg-muted-foreground/60');
</script>

<ModelCard href="#{family.name}" tooltip={family.name}>
	{#snippet avatar()}
		<span class="relative inline-flex">
			<ModelOrgAvatar org={family.org} />

			{#if variant === 'avatar'}
				<span
					class="absolute -right-0.5 -bottom-0.5 size-3 rounded-full ring-2 ring-(--surface-raised) {lamp}"
					title="{label} in Llama"
				></span>
			{/if}
		</span>
	{/snippet}

	{#snippet title()}
		<span class="flex min-w-0 flex-col gap-0.5">
			<span class="truncate font-medium">{family.name}</span>

			<span class="flex min-w-0 items-center gap-1.5 font-mono text-caption text-muted-foreground tabular-nums">
				<span class="truncate">{sizesSummary(family)}</span>

				{#if variant === 'line'}
					<span aria-hidden="true">·</span>
					<span class="inline-flex shrink-0 items-center gap-1 font-sans {state === 'loaded' ? 'text-fg-success' : ''}">
						<span class="size-1.5 rounded-full {lamp}"></span>
						{label}
					</span>
				{/if}
			</span>
		</span>
	{/snippet}

	{#snippet actions()}
		<span class="flex h-9 items-center gap-2 pr-1.5">
			{#if variant === 'corner'}
				<span
					class="inline-flex {state === 'loaded' ? 'text-fg-success' : 'text-foreground'}"
					title="{label} in Llama"
				>
					<CircleCheck class="size-icon-sm" />
				</span>
				<span class="h-3 w-px bg-border"></span>
			{/if}

			<ModelCapabilityIcons
				modalities={modalitiesOf(family.sizes)}
				supportsThinking={reasons(family.sizes)}
				supportsToolUse={usesTools(family.sizes)}
			/>
		</span>
	{/snippet}

	<div class="line-clamp-2 text-body-sm text-pretty text-muted-foreground">
		{family.description.replaceAll('**', '')}
	</div>

	{#snippet footer()}
		<span class="flex items-center gap-3 whitespace-nowrap tabular-nums">
			{#if range}
				<span class="inline-flex items-center gap-1"><Database class="size-icon-xs" />{formatSizeRange(range.min, range.max)}</span>
			{/if}
			{#if mem}
				<span class="inline-flex items-center gap-1"><MemoryStick class="size-icon-xs" />{mem} GB+</span>
			{/if}
		</span>
	{/snippet}

	{#snippet footerEnd()}
		{#if variant === 'badge'}
			<Badge variant={state === 'loaded' ? 'success' : 'secondary'}>{label}</Badge>
		{:else}
			<span class="text-muted-foreground tabular-nums">{releasedFor(family)}</span>
		{/if}
	{/snippet}
</ModelCard>
