<script lang="ts">
	import ModelAvatar from '../ModelAvatar.svelte';
	import ModelCapabilities from '../ModelCapabilities.svelte';
	import ModelContext from '../ModelContext.svelte';
	import ModelId from '../ModelId.svelte';
	import type { ModelQuantGroup } from './utils';
	import { configuredContext } from './utils';
	import { ChevronDown, ChevronUp } from '@lucide/svelte';
	import { MODEL_ROW_GRID_CLASS, SETTINGS_KEYS } from '$lib/constants';
	import { ModelGroupKind } from '$lib/enums';
	import { modelsStore, settingsStore } from '$lib/stores';

	interface Props {
		entry: ModelQuantGroup;
		expanded: boolean;
		onToggle: () => void;
		/** Left padding in px, from the nesting depth. */
		indent?: number;
	}

	let { entry, expanded, indent = 0, onToggle }: Props = $props();

	let groupLabel = $derived(
		entry.kind === ModelGroupKind.VARIANTS
			? `${entry.quants.length} variants`
			: `${entry.quants.length} quants available`
	);
	let anyLoaded = $derived(entry.quants.some((quant) => modelsStore.isModelRunning(quant.model)));
	// a repo row stands for its quants, so it reports what they agree on
	let contextSource = $derived(entry.quants.find((quant) => quant.contextLength) ?? entry.base);
	let mediaSource = $derived(entry.quants.find((quant) => quant.modalities) ?? entry.base);
</script>

<!-- the name cell discloses the quants, so the row itself takes no focus -->
<div class={[MODEL_ROW_GRID_CLASS, 'group rounded-md px-2 py-2.5 transition hover:bg-muted/40']}>
	<button
		aria-expanded={expanded}
		class="flex min-w-0 cursor-pointer items-center gap-3 rounded-md text-left outline-none focus-visible:ring-2 focus-visible:ring-ring"
		onclick={onToggle}
		style="padding-left: {indent}px"
		type="button"
	>
		<ModelAvatar
			option={entry.base}
			showBaseModelAvatar={!settingsStore.config[SETTINGS_KEYS.GROUP_MODELS_BY_FAMILY]}
			showRepoOrgAvatar={settingsStore.config[SETTINGS_KEYS.GROUP_MODELS_BY_FAMILY]}
			size="size-9"
		/>

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

				<ModelCapabilities option={entry.base} />
			</span>

			<span class="block text-xs text-muted-foreground">{groupLabel}</span>
		</span>
	</button>

	<ModelContext
		class="justify-self-end"
		configured={configuredContext(contextSource)}
		option={contextSource}
	/>

	<span class="justify-self-center">
		<span
			class="block h-2.5 w-2.5 rounded-full {anyLoaded
				? 'bg-emerald-500'
				: 'border border-muted-foreground/50'}"
		></span>
	</span>

	<span class="flex justify-center">
		{#if expanded}
			<ChevronUp class="h-3.5 w-3.5 text-muted-foreground" />
		{:else}
			<ChevronDown class="h-3.5 w-3.5 text-muted-foreground" />
		{/if}
	</span>
</div>
