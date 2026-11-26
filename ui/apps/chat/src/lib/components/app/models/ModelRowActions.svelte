<script lang="ts">
	import { Heart, HeartOff, Info, Mic } from '@lucide/svelte';
	import { ActionIcon } from '#lib/components/app/index.ts';
	import { VOICE_DICTATION_MODEL_LABEL } from '#lib/constants/index.ts';
	import { modelsStore, serverStore, uiStore, voiceStore } from '#lib/stores/index.ts';
	import type { ModelOption } from '#lib/types/models.js';

	interface Props {
		isFav: boolean;
		option: ModelOption;
		/** Selector rows reveal the actions on hover, table rows keep them visible. */
		revealOnHover?: boolean;
	}

	let { isFav, option, revealOnHover = true }: Props = $props();

	// the favorite heart and the dictation model's microphone stay visible, so the reveal
	// applies to the other icons
	// any router model that hears can be the one that transcribes dictation
	let canDictate = $derived(serverStore.isRouterMode && voiceStore.roleOf(option.model) !== null);
	let dictates = $derived(voiceStore.isDictationModel(option.model));
	let dictationLabel = $derived.by(() => {
		if (!dictates) return VOICE_DICTATION_MODEL_LABEL.use;

		return voiceStore.dictationIsAutomatic
			? VOICE_DICTATION_MODEL_LABEL.automatic
			: VOICE_DICTATION_MODEL_LABEL.chosen;
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

	{#if canDictate}
		<!-- like the heart: lit on the model that dictates, among the hover actions on the others -->
		<span
			class={!dictates && revealOnHover ? revealClass : ''}
			data-dictates={dictates || undefined}
			data-slot="model-dictation-action"
		>
			<ActionIcon
				class={dictates
					? 'size-icon-lg text-destructive hover:text-destructive'
					: 'size-icon-lg hover:text-foreground'}
				icon={Mic}
				iconSize="size-icon-md"
				onclick={() => voiceStore.toggleDictation(option.model)}
				tooltip={dictationLabel}
				tooltipAsTitle
			/>
		</span>
	{/if}

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
