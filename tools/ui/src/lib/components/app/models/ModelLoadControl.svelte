<script lang="ts">
	import { CircleAlert, Cloud, Loader2, Power, RotateCw, Upload } from '@lucide/svelte';
	import { ActionIcon } from '$lib/components/app';
	import { BackendIcon } from '$lib/components/app/backends';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { modelsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { getBackend } from '$lib/utils/api-base';

	interface Props {
		/** Backend can load and unload models, llama-compat servers only. */
		canLoad: boolean;
		isFailed?: boolean;
		isLoaded: boolean;
		isLoading?: boolean;
		isSleeping?: boolean;
		option: ModelOption;
		/** Table rows keep the load action visible, selector rows reveal it on hover. */
		revealOnHover?: boolean;
		/** Non-loadable rows show the provider mark, which identifies them in a flat list. */
		showBackendMark?: boolean;
		/** Table rows mark a remote provider, which this UI cannot load or unload. */
		showRemoteMark?: boolean;
	}

	let {
		canLoad,
		isFailed = false,
		isLoaded,
		isLoading = false,
		isSleeping = false,
		option,
		revealOnHover = true,
		showBackendMark = false,
		showRemoteMark = false
	}: Props = $props();
</script>

<div class="flex w-5 shrink-0 items-center justify-center">
	{#if !canLoad}
		{#if showBackendMark}
			<BackendIcon backend={getBackend(option.backendId)} class="h-3.5 w-3.5" />
		{:else if showRemoteMark}
			<span class="text-muted-foreground" title="Served remotely"
				><Cloud class="h-3.5 w-3.5" /></span
			>
		{/if}
	{:else if isLoading}
		<Loader2 class="{ICON_CLASS_DEFAULT} animate-spin text-muted-foreground" />
	{:else}
		<!-- the state dot is what the row shows at rest; the action takes its place on hover -->
		{#if revealOnHover}
			{#if isFailed}
				<CircleAlert
					class="h-3.5 w-3.5 text-red-500 group-hover:hidden [@media(pointer:coarse)]:hidden"
				/>
			{:else}
				<span
					class="h-2 w-2 rounded-full group-hover:hidden [@media(pointer:coarse)]:hidden {isSleeping
						? 'bg-orange-400'
						: isLoaded
							? 'bg-green-500'
							: 'bg-muted-foreground/50'}"
				></span>
			{/if}
		{/if}

		<div class={revealOnHover ? 'hidden group-hover:flex [@media(pointer:coarse)]:flex' : 'flex'}>
			{#if isFailed}
				<ActionIcon
					class="h-5 w-5 text-red-500 hover:text-foreground"
					icon={RotateCw}
					iconSize="h-4 w-4"
					onclick={() => modelsStore.status.load(option.model)}
					stopPropagationOnClick
					tooltip="Retry loading model"
					tooltipAsTitle
				/>
			{:else if isLoaded || isSleeping}
				<ActionIcon
					class="h-5 w-5 hover:text-foreground"
					icon={Upload}
					iconSize="h-4 w-4"
					onclick={() => modelsStore.status.unload(option.model)}
					stopPropagationOnClick
					tooltip="Unload model"
					tooltipAsTitle
				/>
			{:else}
				<ActionIcon
					class="h-5 w-5 hover:text-foreground"
					icon={Power}
					iconSize="h-4 w-4"
					onclick={() => modelsStore.status.load(option.model)}
					stopPropagationOnClick
					tooltip="Load model"
					tooltipAsTitle
				/>
			{/if}
		</div>
	{/if}
</div>
