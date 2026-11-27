<!--
	A model's name with its badges (draft kind, parameters, quantization, drafts, aliases, tags) and its capability,
	context and size icons. The app passes the parts it already knows (a parsed id, a catalog entry) and the user's
	display settings as the props it sets or leaves out, so the same markup serves Llama Chat and llama.app.
-->
<script lang="ts">
	import ModelCapabilityIcons from './ModelCapabilityIcons.svelte';
	import ModelDraftSidecars from './ModelDraftSidecars.svelte';
	import * as Tooltip from '../ui/tooltip';
	import { Database, ScrollText } from '@lucide/svelte';
	import {
		MODEL_BADGE_CLASS,
		MODEL_TAG_BADGE_CLASS,
		MODEL_VARIANT_BADGE_CLASS
	} from '../../models/constants';
	import { formatParameters, formatSizeRange } from '../../models/format';
	import type { ModelDraftSidecarBadge, ModelModalities, ModelSizeRange } from '../../models/types';

	interface Props {
		/** The model's name, e.g. `gemma-4` or `Gemma 4 E4B`. */
		name: string;
		/** The org or author, shown before the name as `org/name` unless `hideOrgName`. */
		org?: string;
		/** Total parameters, e.g. `26B`. */
		params?: string;
		/** Parameters active per token in a mixture of experts, e.g. `A4B`; joined to `params` as `26B-A4B`. */
		activatedParams?: string;
		/** Quantization, e.g. `Q4_K_M`. */
		quantization?: string;
		/** The draft kind the model itself is, e.g. `mtp` for an MTP head; badged first. */
		sidecar?: string;
		/** Draft sidecar kinds the model comes with, e.g. `mtp`, one badge each and no quant. */
		draftKinds?: string[];
		/** Draft sidecars the model comes with, badged with their own quant. */
		draftSidecars?: ModelDraftSidecarBadge[];
		/**
		 * Names the model is served under. One alias replaces the name, which moves to a badge; more than one are
		 * badged after the name, two at most and then `+x more`.
		 */
		aliases?: string[];
		tags?: string[];
		modalities?: ModelModalities;
		supportsThinking?: boolean;
		supportsToolUse?: boolean;
		/** Context length in tokens; renders a context icon when set. */
		contextLength?: number;
		/** Min/max file size across quants, in bytes; renders a size icon when set. */
		sizeRange?: ModelSizeRange | null;
		hideName?: boolean;
		hideOrgName?: boolean;
		hideCapabilities?: boolean;
		hideModalities?: boolean;
		/** Render the capability/modality/context icons on a second row. */
		iconsOnNewLine?: boolean;
		/** Give the name a row of its own, with its badges and icons together under it. */
		stackId?: boolean;
		/** Allow badges to wrap onto new lines instead of truncating. */
		wrap?: boolean;
		/** Native title for the root element; keeps long lists light where a floating tooltip per row is too costly. */
		title?: string;
		/** Shown in a tooltip over the name and badges, e.g. the raw id the name was parsed from. */
		tooltip?: string;
		class?: string;
	}

	let {
		activatedParams,
		aliases = [],
		class: className = '',
		contextLength,
		draftKinds = [],
		draftSidecars = [],
		hideCapabilities = false,
		hideModalities = false,
		hideName = false,
		hideOrgName = false,
		iconsOnNewLine = false,
		modalities,
		name,
		org,
		params,
		quantization,
		sidecar,
		sizeRange,
		stackId = false,
		supportsThinking = false,
		supportsToolUse = false,
		tags = [],
		title,
		tooltip,
		wrap = false,
		...rest
	}: Props = $props();

	/** Alias badges beyond this many collapse into a single `+x more` badge. */
	const MAX_ALIAS_BADGES = 2;

	let uniqueAliases = $derived([...new Set(aliases)]);
	let uniqueTags = $derived([...new Set(tags)]);
	// the model itself can already be one of the kinds
	let uniqueDraftKinds = $derived([...new Set(draftKinds.filter((kind) => kind !== sidecar))]);

	let primaryAlias = $derived(uniqueAliases.length === 1 ? uniqueAliases[0] : null);
	let displayName = $derived(primaryAlias ?? name);

	let hasBadges = $derived(
		sidecar ||
			uniqueDraftKinds.length > 0 ||
			draftSidecars.length > 0 ||
			params ||
			quantization ||
			primaryAlias ||
			uniqueAliases.length > 1 ||
			uniqueTags.length > 0
	);
</script>

{#snippet nameAndBadges()}
	{#if !hideName}
		<span class="min-w-0 truncate font-medium {stackId ? 'basis-full' : ''}">
			{#if !hideOrgName && org}{org}/{/if}{displayName}
		</span>
	{/if}

	{#if hasBadges}
		<!-- the badges keep their width, so a long name truncates around them instead of
		     a label being cut in half -->
		<span
			class="inline-flex min-w-0 items-center gap-1 overflow-hidden {wrap
				? 'flex-wrap'
				: 'shrink-0'}"
		>
			{#if sidecar}
				<span class={MODEL_VARIANT_BADGE_CLASS} title={`${sidecar.toUpperCase()} draft model`}>
					{sidecar}
				</span>
			{/if}

			{#if params}
				<span class={MODEL_BADGE_CLASS}>
					{params}{activatedParams ? `-${activatedParams}` : ''}
				</span>
			{/if}

			{#if quantization}
				<span class={MODEL_BADGE_CLASS}>{quantization}</span>
			{/if}

			{#each uniqueDraftKinds as kind (kind)}
				<span class={MODEL_VARIANT_BADGE_CLASS} title="{kind.toUpperCase()} draft model">
					{kind}
				</span>
			{/each}

			<ModelDraftSidecars {draftSidecars} />

			{#if primaryAlias}
				{#if primaryAlias !== name}
					<span class="{MODEL_BADGE_CLASS} max-w-32 truncate" title={name}>{name}</span>
				{/if}
			{:else if uniqueAliases.length > 1}
				{#each uniqueAliases.slice(0, MAX_ALIAS_BADGES) as alias (alias)}
					<span class="{MODEL_BADGE_CLASS} max-w-32 truncate" title={alias}>{alias}</span>
				{/each}

				{#if uniqueAliases.length > MAX_ALIAS_BADGES}
					<span class={MODEL_BADGE_CLASS} title={uniqueAliases.slice(MAX_ALIAS_BADGES).join(', ')}>
						+{uniqueAliases.length - MAX_ALIAS_BADGES} more
					</span>
				{/if}
			{/if}

			{#each uniqueTags as tag (tag)}
				<span class={MODEL_TAG_BADGE_CLASS}>{tag}</span>
			{/each}
		</span>
	{/if}
{/snippet}

<!-- badges, tags and icons do not shrink, so overflow-hidden bounds the name -->
<span
	class="flex min-w-0 items-center gap-1.5 overflow-hidden {wrap ? 'flex-wrap' : ''} {iconsOnNewLine
		? 'flex-col items-start'
		: ''} {className}"
	{title}
	{...rest}
>
	<span
		class="flex min-w-0 items-center gap-1.5 max-md:gap-0.5 {wrap || stackId ? 'flex-wrap' : ''}"
	>
		{#if tooltip}
			<Tooltip.Root>
				<Tooltip.Trigger class="flex min-w-0 items-center gap-1.5">
					{@render nameAndBadges()}
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>{tooltip}</p>
				</Tooltip.Content>
			</Tooltip.Root>
		{:else}
			{@render nameAndBadges()}
		{/if}

		{#if !iconsOnNewLine}
			<ModelCapabilityIcons
				{hideCapabilities}
				{hideModalities}
				{modalities}
				{supportsThinking}
				{supportsToolUse}
			/>
		{/if}
	</span>

	{#if iconsOnNewLine || contextLength || sizeRange}
		<span class="inline-flex items-center gap-1.5">
			{#if iconsOnNewLine}
				<ModelCapabilityIcons
					{hideCapabilities}
					{hideModalities}
					{modalities}
					{supportsThinking}
					{supportsToolUse}
				/>
			{/if}

			{#if contextLength}
				<span class="inline-flex items-center gap-1 text-muted-foreground">
					<ScrollText class="size-icon-xs" />

					<span class="text-caption">{formatParameters(contextLength)}</span>
				</span>
			{/if}

			{#if sizeRange}
				<span class="inline-flex items-center gap-1 text-muted-foreground">
					<Database class="size-icon-xs" />

					<span class="text-caption">{formatSizeRange(sizeRange.min, sizeRange.max)}</span>
				</span>
			{/if}
		</span>
	{/if}
</span>
