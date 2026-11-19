<script lang="ts">
	import { Heart, HeartOff, Info, Mic } from '@lucide/svelte';
	import { ActionIcon } from '#lib/components/app/index.ts';
	import { VoiceModelRole } from '#lib/enums/index.ts';
	import { modelsStore, serverStore, uiStore, voiceStore } from '#lib/stores/index.ts';
	import type { ModelOption } from '#lib/types/models.js';

	interface Props {
		isFav: boolean;
		option: ModelOption;
		/** Selector rows reveal the actions on hover, table rows keep them visible. */
		revealOnHover?: boolean;
	}

	let { isFav, option, revealOnHover = true }: Props = $props();

	// the favorite heart stays visible, so the reveal applies to the other icons
	// a router model that hears, other than the one that dictates, can be made the one
	let canDictate = $derived.by(() => {
		const role = voiceStore.roleOf(option.model);

		return serverStore.isRouterMode && role !== null && role !== VoiceModelRole.DICTATION;
	});

	const revealClass =
		'pointer-events-none opacity-0 group-hover:pointer-events-auto group-hover:opacity-100 [@media(pointer:coarse)]:pointer-events-auto [@media(pointer:coarse)]:opacity-100';
</script>

<div
	class="flex items-center justify-center gap-1 max-md:gap-2.5"
	onclick={(event) => event.stopPropagation()}
	onkeydown={(event) => event.stopPropagation()}
	role="presentation"
>
	{#if canDictate}
		<span class={revealOnHover ? revealClass : ''}>
			<ActionIcon
				class="size-icon-lg hover:text-foreground"
				icon={Mic}
				iconSize="size-icon-md"
				onclick={() => voiceStore.useForDictation(option.model)}
				tooltip="Use for dictation"
				tooltipAsTitle
			/>
		</span>
	{/if}

	<span class={revealOnHover ? revealClass : ''}>
		<ActionIcon
			class="size-icon-lg hover:text-foreground"
			icon={Info}
			iconSize="size-icon-md"
			onclick={() => uiStore.openModelDetails(option.id)}
			tooltip="Manage model"
			tooltipAsTitle
		/>
	</span>

	{#if isFav}
		<span class="flex size-5 items-center justify-center">
			<span class="flex group-hover:hidden [@media(pointer:coarse)]:hidden">
				<ActionIcon
					class="size-icon-lg text-rose-500"
					icon={Heart}
					iconSize="size-icon-md"
					onclick={() => modelsStore.toggleFavorite(option.model)}
					tooltip="Remove from favorites"
					tooltipAsTitle
				/>
			</span>

			<span class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
				<ActionIcon
					class="size-icon-lg hover:text-foreground"
					icon={HeartOff}
					iconSize="size-icon-md"
					onclick={() => modelsStore.toggleFavorite(option.model)}
					tooltip="Remove from favorites"
					tooltipAsTitle
				/>
			</span>
		</span>
	{:else}
		<span class={revealOnHover ? revealClass : ''}>
			<ActionIcon
				class="size-icon-lg hover:text-foreground"
				icon={Heart}
				iconSize="size-icon-md"
				onclick={() => modelsStore.toggleFavorite(option.model)}
				tooltip="Add to favorites"
				tooltipAsTitle
			/>
		</span>
	{/if}
</div>
