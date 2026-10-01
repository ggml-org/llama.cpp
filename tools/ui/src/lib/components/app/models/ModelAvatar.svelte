<script lang="ts">
	import ModelOrgAvatar from './ModelOrgAvatar.svelte';
	import { HF_BASE_MODEL_TAG_REGEX } from '$lib/constants';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import type { ModelOption } from '$lib/types/models';
	import { nearViewport, orgOf } from '$lib/utils';
	import type { Snippet } from 'svelte';

	interface Props {
		class?: string;
		/** Rendered when the model has neither an org avatar nor a provider mark. */
		fallback?: Snippet;
		option: ModelOption;
		quantPositionClass?: string;
		quantSize?: string;
		/** Show the base model's org as the main image, the repo (quantizer) org as the
		 *  corner badge. The base org costs one Hub request per repo. */
		showBaseModelAvatar?: boolean;
		/** Show the repo's own org as the main image, skipping the base model.
		 *  Used inside a heading that already carries the base org. */
		showRepoOrgAvatar?: boolean;
		/** Keep the quantizer badge off, e.g. when a row stands for a whole family. */
		showQuantBadge?: boolean;
		size?: string;
	}

	let {
		class: className = '',
		fallback,
		option,
		quantPositionClass = '-bottom-1 -right-1',
		quantSize = 'h-3 w-3',
		showBaseModelAvatar = false,
		showQuantBadge = true,
		showRepoOrgAvatar = false,
		size = 'size-5'
	}: Props = $props();

	let parsedId = $derived(ModelsService.parseModelId(option.model));
	let orgName = $derived(parsedId.orgName);
	// Avatars come from the Hub, so with the metadata setting off they are
	// hidden entirely and no base-model lookup runs.
	let hubEnabled = $derived(HuggingFaceService.isEnabled());
	let tagBaseModel = $derived(
		(option.tags ?? [])
			.find((t) => HF_BASE_MODEL_TAG_REGEX.test(t))
			?.match(HF_BASE_MODEL_TAG_REGEX)?.[1] ?? null
	);
	// long lists mount hundreds of avatars at once, so the base org lookup waits
	// until the row is near the viewport
	let fetchedBaseModelOrg = $state<string | null>(null);
	let baseModelOrg = $derived(orgOf(tagBaseModel) || fetchedBaseModelOrg);
	let isNearViewport = $state(false);

	$effect(() => {
		fetchedBaseModelOrg = null;

		if (!isNearViewport || !showBaseModelAvatar || !orgName || tagBaseModel) return;

		if (!hubEnabled) return;

		let cancelled = false;

		void HuggingFaceService.getBaseModel(option.model)
			.then((base) => {
				if (!cancelled && base?.org) fetchedBaseModelOrg = base.org;
			})
			// best-effort lookup: offline or unknown repos keep the repo org
			.catch(() => {});

		return () => {
			cancelled = true;
		};
	});
</script>

{#if orgName && hubEnabled}
	<span
		use:nearViewport={() => (isNearViewport = true)}
		class={['inline-flex shrink-0', className]}
	>
		<ModelOrgAvatar
			class="mt-0"
			org={showRepoOrgAvatar ? orgName : (baseModelOrg ?? orgName)}
			quantOrg={showBaseModelAvatar && showQuantBadge && !showRepoOrgAvatar ? orgName : undefined}
			{quantPositionClass}
			{quantSize}
			{size}
		/>
	</span>
{:else}
	{@render fallback?.()}
{/if}
