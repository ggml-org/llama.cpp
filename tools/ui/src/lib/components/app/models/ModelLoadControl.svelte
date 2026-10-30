<script lang="ts">
	import { ActionIcon } from '$lib/components/app';
	import { ICON_CLASS_DEFAULT, MODEL_LOAD_ICONS } from '$lib/constants';
	import { modelsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		class?: string;
		isFailed?: boolean;
		isLoaded: boolean;
		isLoading?: boolean;
		isSleeping?: boolean;
		option: ModelOption;
		/** Table rows keep the load action visible, selector rows reveal it on hover. */
		revealOnHover?: boolean;
		/** Renders the state alone, for a caller that moves load and unload elsewhere. */
		showAction?: boolean;
	}

	let {
		class: className = '',
		isFailed = false,
		isLoaded,
		isLoading = false,
		isSleeping = false,
		option,
		revealOnHover = true,
		showAction = true
	}: Props = $props();
</script>

<div class={['flex w-5 shrink-0 items-center justify-center', className]}>
	{#if isLoading}
		<MODEL_LOAD_ICONS.loading class="{ICON_CLASS_DEFAULT} animate-spin text-muted-foreground" />
	{:else}
		<!-- the state dot is what the row shows at rest; the action takes its place on hover -->
		{#if showAction && revealOnHover}
			{#if isFailed}
				<MODEL_LOAD_ICONS.failed
					class="size-icon-sm text-status-danger group-hover:hidden [@media(pointer:coarse)]:hidden"
				/>
			{:else}
				<span
					class="size-2 rounded-full group-hover:hidden [@media(pointer:coarse)]:hidden {isSleeping
						? 'status-lamp orb-warning'
						: isLoaded
							? 'status-lamp orb-success'
							: 'bg-muted-foreground/50'}"
				></span>
			{/if}
		{:else}
			{#if isFailed}
				<MODEL_LOAD_ICONS.failed class="size-icon-sm text-status-danger" />
			{:else}
				<span
					class="size-2 rounded-full {isSleeping
						? 'status-lamp orb-warning'
						: isLoaded
							? 'status-lamp orb-success'
							: 'bg-muted-foreground/50'}"
				></span>
			{/if}
		{/if}

		<!-- the wrapper has to pick one display value: flex and hidden together
		     leave the winner to stylesheet order -->
		<div
			class={!showAction
				? 'hidden'
				: revealOnHover
					? 'hidden group-hover:flex [@media(pointer:coarse)]:flex'
					: 'flex'}
		>
			{#if isFailed}
				<ActionIcon
					class="size-icon-lg text-red-500 hover:text-foreground"
					icon={MODEL_LOAD_ICONS.retry}
					iconSize="size-icon-md"
					onclick={() => modelsStore.status.load(option.model)}
					stopPropagationOnClick
					tooltip="Retry loading model"
					tooltipAsTitle
				/>
			{:else if isLoaded || isSleeping}
				<ActionIcon
					class="size-icon-lg hover:text-foreground"
					icon={MODEL_LOAD_ICONS.unload}
					iconSize="size-icon-md"
					onclick={() => modelsStore.status.unload(option.model)}
					stopPropagationOnClick
					tooltip="Unload model"
					tooltipAsTitle
				/>
			{:else}
				<ActionIcon
					class="size-icon-lg hover:text-foreground"
					icon={MODEL_LOAD_ICONS.load}
					iconSize="size-icon-md"
					onclick={() => modelsStore.status.load(option.model)}
					stopPropagationOnClick
					tooltip="Load model"
					tooltipAsTitle
				/>
			{/if}
		</div>
	{/if}
</div>
