<!--
	A model's name with its badges (parameters, quantization, drafts, tags) and its capability, context and size icons.
	The props-in version of ui/apps/chat's ModelId: the app passes the parts it already knows (a parsed id, a catalog
	entry) instead of a raw id read against the user's settings, so the same markup serves Llama Chat and llama.app.
-->
<script lang="ts">
	import ModelCapabilityIcons from './ModelCapabilityIcons.svelte';
	import { Database, ScrollText } from '@lucide/svelte';
	import {
		MODEL_BADGE_CLASS,
		MODEL_TAG_BADGE_CLASS,
		MODEL_VARIANT_BADGE_CLASS
	} from '../../models/constants';
	import { formatParameters, formatSizeRange } from '../../models/format';
	import type { ModelModalities, ModelSizeRange } from '../../models/types';

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
		/** Draft sidecar kinds the model comes with, e.g. `mtp`, one badge each. */
		draftKinds?: string[];
		tags?: string[];
		modalities?: ModelModalities;
		supportsThinking?: boolean;
		supportsToolUse?: boolean;
		/** Context length in tokens; renders a context icon when set. */
		contextLength?: number;
		/** Min/max file size across quants, in bytes; renders a size icon when set. */
		sizeRange?: ModelSizeRange | null;
		hideOrgName?: boolean;
		hideCapabilities?: boolean;
		hideModalities?: boolean;
		/** Render the capability/modality/context icons on a second row. */
		iconsOnNewLine?: boolean;
		/** Give the name a row of its own, with its badges and icons together under it. */
		stackId?: boolean;
		/** Allow badges to wrap onto new lines instead of truncating. */
		wrap?: boolean;
		/** Native title for the root element. */
		title?: string;
		class?: string;
	}

	let {
		activatedParams,
		class: className = '',
		contextLength,
		draftKinds = [],
		hideCapabilities = false,
		hideModalities = false,
		hideOrgName = false,
		iconsOnNewLine = false,
		modalities,
		name,
		org,
		params,
		quantization,
		sizeRange,
		stackId = false,
		supportsThinking = false,
		supportsToolUse = false,
		tags = [],
		title,
		wrap = false
	}: Props = $props();

	let uniqueTags = $derived([...new Set(tags)]);
	let uniqueDraftKinds = $derived([...new Set(draftKinds)]);

	let hasBadges = $derived(
		uniqueDraftKinds.length > 0 || params || quantization || uniqueTags.length > 0
	);
</script>

<!-- badges, tags and icons do not shrink, so overflow-hidden bounds the name -->
<span
	class="flex min-w-0 items-center gap-1.5 overflow-hidden {wrap ? 'flex-wrap' : ''} {iconsOnNewLine
		? 'flex-col items-start'
		: ''} {className}"
	{title}
>
	<span
		class="flex min-w-0 items-center gap-1.5 max-md:gap-0.5 {wrap || stackId ? 'flex-wrap' : ''}"
	>
		<span class="min-w-0 truncate font-medium {stackId ? 'basis-full' : ''}">
			{#if !hideOrgName && org}{org}/{/if}{name}
		</span>

		{#if hasBadges}
			<!-- the badges keep their width, so a long name truncates around them instead of
			     a label being cut in half -->
			<span
				class="inline-flex min-w-0 items-center gap-1 overflow-hidden {wrap
					? 'flex-wrap'
					: 'shrink-0'}"
			>
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

				{#each uniqueTags as tag (tag)}
					<span class={MODEL_TAG_BADGE_CLASS}>{tag}</span>
				{/each}
			</span>
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

					<span class="text-xs">{formatParameters(contextLength)}</span>
				</span>
			{/if}

			{#if sizeRange}
				<span class="inline-flex items-center gap-1 text-muted-foreground">
					<Database class="size-icon-xs" />

					<span class="text-xs">{formatSizeRange(sizeRange.min, sizeRange.max)}</span>
				</span>
			{/if}
		</span>
	{/if}
</span>
