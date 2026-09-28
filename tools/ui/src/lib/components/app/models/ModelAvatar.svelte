<script lang="ts">
	import ModelsDiscoverAvatar from './discover/ModelsDiscoverAvatar.svelte';
	import { BackendIcon } from '$lib/components/app/backends';
	import { Logo } from '$lib/components/app/misc';
	import { HF_BASE_MODEL_TAG_REGEX, LOCAL_BACKEND_ID, MODEL_ICON } from '$lib/constants';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import type { ModelOption } from '$lib/types/models';
	import { orgOf } from '$lib/utils';
	import { getBackend } from '$lib/utils/api-base';
	import { getBackendCapabilities } from '$lib/utils/backend';
	import type { Snippet } from 'svelte';

	interface Props {
		class?: string;
		/** Rendered when the model has neither an org avatar nor a provider mark. */
		fallback?: Snippet;
		option: ModelOption;
		quantPositionClass?: string;
		quantSize?: string;
		/** Show the base model's org as the main image and the repo (quantizer) org as the
		 *  corner badge. Resolving the base org costs one Hugging Face request per repo, so
		 *  it waits until the row is near the viewport. */
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
	// a llama-compat model whose id carries no `org/name` is not a Hugging Face repo,
	// so the provider's own mark identifies it better than an initial
	let isLlamaCompat = $derived(getBackendCapabilities(getBackend(option.backendId)).props);
	let useProviderIcon = $derived(isLlamaCompat && !orgName);
	// the bundled server has no favicon to resolve, its mark is the llama.cpp logo
	let isLocal = $derived(getBackend(option.backendId)?.id === LOCAL_BACKEND_ID);
	let tagBaseModel = $derived(
		(option.tags ?? [])
			.find((t) => HF_BASE_MODEL_TAG_REGEX.test(t))
			?.match(HF_BASE_MODEL_TAG_REGEX)?.[1] ?? null
	);
	let fetchedBaseModelOrg = $state<string | null>(null);
	let baseModelOrg = $derived(orgOf(tagBaseModel) || fetchedBaseModelOrg);
	let avatarEl = $state<HTMLElement | null>(null);
	let isNearViewport = $state(false);

	// Long lists mount hundreds of avatars at once; resolving every base model up
	// front means one request per row, so wait until a row is actually near the viewport.
	$effect(() => {
		if (isNearViewport || !avatarEl) return;

		if (typeof IntersectionObserver === 'undefined') {
			isNearViewport = true;

			return;
		}

		const observer = new IntersectionObserver(
			(entries) => {
				if (entries.some((entry) => entry.isIntersecting)) {
					isNearViewport = true;
					observer.disconnect();
				}
			},
			{ rootMargin: '200px' }
		);

		observer.observe(avatarEl);

		return () => observer.disconnect();
	});

	$effect(() => {
		fetchedBaseModelOrg = null;

		if (!isNearViewport || !showBaseModelAvatar || !orgName || tagBaseModel) return;

		// external provider ids (`~openai/gpt-...`, `deepseek/deepseek-chat`) are
		// not HF repos; their org is already the provider slug
		const backend = getBackend(option.backendId);

		if (backend && !getBackendCapabilities(backend).props) return;

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

{#if useProviderIcon}
	<span class={['inline-flex shrink-0', className]}>
		<BackendIcon backend={getBackend(option.backendId)} class={size}>
			{#snippet fallback()}
				{#if isLocal}
					<Logo class={size} style="--size: 100%" />
				{:else}
					<MODEL_ICON class={size} />
				{/if}
			{/snippet}
		</BackendIcon>
	</span>
{:else if orgName}
	<span bind:this={avatarEl} class={['inline-flex shrink-0', className]}>
		<ModelsDiscoverAvatar
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
