<script lang="ts">
	import ModelAvatar from '../ModelAvatar.svelte';
	import ModelCapabilities from '../ModelCapabilities.svelte';
	import ModelContext from '../ModelContext.svelte';
	import ModelId from '../ModelId.svelte';
	import type { ModelQuantGroup } from './utils';
	import { configuredContext, type ModelOverride } from './utils';
	import { ChevronDown, ChevronUp } from '@lucide/svelte';
	import { MODEL_ROW_GRID_CLASS, MODEL_ROW_TRAILING_CELL_CLASS } from '$lib/constants';
	import { KeyboardKey, ModelGroupKind } from '$lib/enums';
	import { deviceStore, modelsStore } from '$lib/stores';

	interface Props {
		entry: ModelQuantGroup;
		expanded: boolean;
		onToggle: () => void;
		/** Stored per-model overrides, for the drafts and context the row reports. */
		overrides?: Record<string, ModelOverride>;
		/** Left padding in px, from the nesting depth. */
		indent?: number;
	}

	let { entry, expanded, indent = 0, onToggle, overrides }: Props = $props();

	let providerCount = $derived(new Set(entry.quants.map((option) => option.backendId ?? '')).size);
	let groupLabel = $derived(
		entry.kind === ModelGroupKind.PROVIDERS
			? `${providerCount} provider${providerCount === 1 ? '' : 's'}`
			: entry.kind === ModelGroupKind.VARIANTS
				? `${entry.quants.length} variants`
				: `${entry.quants.length} quants available`
	);
	let anyLoaded = $derived(entry.quants.some((quant) => modelsStore.isModelRunning(quant.model)));
	// a repo row stands for its quants, so it reports what they agree on
	let contextSource = $derived(entry.quants.find((quant) => quant.contextLength) ?? entry.base);
	let mediaSource = $derived(entry.quants.find((quant) => quant.modalities) ?? entry.base);

	function handleKeydown(event: KeyboardEvent): void {
		if (event.key === KeyboardKey.SPACE) event.preventDefault();

		if (event.key === KeyboardKey.ENTER || event.key === KeyboardKey.SPACE) onToggle();
	}
</script>

<div
	class={[
		MODEL_ROW_GRID_CLASS,
		'relative cursor-pointer rounded-md px-2 py-2.5 transition hover:bg-muted/40 max-md:px-3 max-md:py-3.5'
	]}
	onclick={onToggle}
	onkeydown={handleKeydown}
	role="button"
	tabindex="0"
>
	<span class="flex min-w-0 items-center gap-3 max-md:pr-9" style="padding-left: {indent}px">
		<ModelAvatar option={entry.base} size="size-9" />

		<span class="min-w-0">
			<span class="flex min-w-0 items-center gap-1.25">
				<ModelId
					aliases={entry.base.aliases}
					class="min-w-0"
					hideCapabilities
					hideModalities
					hideQuantization
					modalities={mediaSource.modalities}
					modelId={entry.base.model}
					tags={entry.base.tags}
					title={entry.base.model}
				/>

				<ModelCapabilities hideModalities={deviceStore.isMobile} option={entry.base} />
			</span>

			<span class="block text-xs text-muted-foreground">{groupLabel}</span>
		</span>
	</span>

	<ModelContext
		class="justify-self-end max-md:hidden"
		configured={configuredContext(contextSource, overrides)}
		option={contextSource}
	/>

	<span class="justify-self-center max-md:hidden">
		<span
			class="block h-2.5 w-2.5 rounded-full {anyLoaded
				? 'bg-emerald-500'
				: 'border border-muted-foreground/50'}"
		></span>
	</span>

	<span class="flex justify-center {MODEL_ROW_TRAILING_CELL_CLASS}">
		{#if expanded}
			<ChevronUp class="h-3.5 w-3.5 text-muted-foreground" />
		{:else}
			<ChevronDown class="h-3.5 w-3.5 text-muted-foreground" />
		{/if}
	</span>
</div>
