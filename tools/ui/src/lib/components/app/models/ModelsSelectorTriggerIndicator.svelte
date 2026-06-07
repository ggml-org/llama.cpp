<script lang="ts">
	import { ChevronDown, CircleAlert, Loader2 } from '@lucide/svelte';
	import { ServerModelStatus } from '$lib/enums';
	import { modelsStore, routerModels } from '$lib/stores/models.svelte';
	import { getModelLoadPhase } from './utils';

	interface Props {
		// selection-transition hint from the parent (e.g. ms.updating). The indicator also
		// derives loading from the model's own status below, so it survives re-selection,
		// concurrent loads, and out-of-band loads that never set the parent's flag.
		loading: boolean;
		modelId?: string | null;
	}

	let { loading, modelId = null }: Props = $props();

	let model = $derived(modelId ? (routerModels().find((m) => m.id === modelId) ?? null) : null);
	// router reports a crash as value="unloaded" + failed=true (see store.getModelStatus)
	let isFailed = $derived(model?.status?.failed === true);
	// Derive loading from the model's own status, not just the parent's transient flag, which
	// resets when loadModel() short-circuits on a load that is already in progress.
	let isModelLoading = $derived(
		!!modelId &&
			!isFailed &&
			(model?.status?.value === ServerModelStatus.LOADING ||
				modelsStore.isModelOperationInProgress(modelId))
	);
	let showLoading = $derived(loading || isModelLoading);

	let load = $derived(
		showLoading && model
			? { stage: model.status?.stage ?? null, progress: model.status?.progress ?? null }
			: null
	);
	let loadPhase = $derived(getModelLoadPhase(load?.stage));
</script>

{#if showLoading}
	{#if loadPhase}
		{@const Icon = loadPhase.icon}
		<span
			class="flex shrink-0 items-center gap-1"
			title={loadPhase.numeric && load?.progress != null
				? `${loadPhase.label} ${Math.round(load.progress * 100)}%`
				: loadPhase.label}
		>
			<!-- a numeric phase with no % yet (e.g. download before sizes are known) is indeterminate: pulse so it doesn't look frozen -->
			<Icon
				class={[
					'h-3 w-3.5 shrink-0',
					loadPhase.anim,
					!loadPhase.anim && load?.progress == null && 'animate-pulse'
				]}
			/>
			{#if loadPhase.numeric && load?.progress != null}
				<!-- reserve room for "100%": the trigger is right-anchored, so a growing % would shift the name -->
				<span class="w-9 shrink-0 text-center font-mono text-xs tabular-nums">
					{Math.round(load.progress * 100)}%
				</span>
			{/if}
		</span>
	{:else}
		<Loader2 class="h-3 w-3.5 shrink-0 animate-spin" />
	{/if}
{:else if isFailed}
	<span class="flex shrink-0 items-center" title="Failed to load">
		<CircleAlert class="h-3 w-3.5 shrink-0 text-red-500" />
	</span>
{:else}
	<ChevronDown class="h-3 w-3.5 shrink-0" />
{/if}
