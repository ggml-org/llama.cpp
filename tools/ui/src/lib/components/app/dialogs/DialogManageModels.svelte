<script lang="ts">
	import { ArrowLeft } from '@lucide/svelte';
	import ModelsDiscover from '$lib/components/app/models/discover/ModelsDiscover.svelte';
	import ModelsManager from '$lib/components/app/models/ModelsManager/ModelsManager.svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import {
		DISCOVER_ICON,
		MODEL_ICON,
		MODELS_DIALOG_VIEW_FADE_MS,
		SETTINGS_KEYS
	} from '$lib/constants';
	import { ModelsDialogView } from '$lib/enums';
	import { modelsStore, settingsStore, uiStore } from '$lib/stores';
	import { untrack } from 'svelte';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { onOpenChange, open = $bindable(false) }: Props = $props();

	// what the user asked for, and what is actually rendered: a view change fades the
	// current one out, swaps, then fades the next one in
	let view = $state<ModelsDialogView>(ModelsDialogView.MANAGE);
	let shownView = $state<ModelsDialogView>(ModelsDialogView.MANAGE);
	let isSwapping = $state(false);
	/** Set while an entry opens the dialog on an explicit view, so the open reset keeps hands off. */
	let viewRequested = false;

	// the discover view exists only while its setting is on
	let discoverEnabled = $derived(
		Boolean(settingsStore.getConfig(SETTINGS_KEYS.ENABLE_DISCOVER_MODELS))
	);

	$effect(() => {
		const next = view;

		if (next === shownView) return;

		// nothing to fade while the dialog is closed, so the view is set before it opens
		if (!open) {
			untrack(() => (shownView = next));

			return;
		}

		untrack(() => (isSwapping = true));

		const timer = setTimeout(() => {
			untrack(() => {
				shownView = next;
				isSwapping = false;
			});
		}, MODELS_DIALOG_VIEW_FADE_MS);

		return () => clearTimeout(timer);
	});

	let title = $derived(view === ModelsDialogView.DISCOVER ? 'Discover' : 'Models');

	// the sidebar's Discover entry opens this dialog on its Discover view
	$effect(() => {
		if (!uiStore.discoverModelsOpen) return;

		untrack(() => {
			uiStore.discoverModelsOpen = false;
			view = discoverEnabled ? ModelsDialogView.DISCOVER : ModelsDialogView.MANAGE;

			// the dialog may already be open, so no open transition will follow
			if (open) return;

			viewRequested = true;
			handleOpenChange(true);
		});
	});

	// opening always lands on the manager, unless an entry requested the discover view
	$effect(() => {
		if (!open) return;

		untrack(() => {
			if (viewRequested) {
				viewRequested = false;
			} else {
				view = ModelsDialogView.MANAGE;
			}

			// sync point: the table opens on cached state and corrects itself when
			// the list fetch lands; a no-op outside router mode
			void modelsStore.fetchRouterModels();
		});
	});

	// the discover view disappears with its setting; fall back to the manager
	$effect(() => {
		if (!discoverEnabled && view === ModelsDialogView.DISCOVER) {
			untrack(() => (view = ModelsDialogView.MANAGE));
		}
	});

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}
</script>

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content
		class="max-md:h-[100dvh]! max-md:w-screen! max-md:max-w-none! max-md:rounded-none! max-md:border-0! md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-380! flex flex-col p-4 pb-0"
		onOpenAutoFocus={(event) => event.preventDefault()}
	>
		<Dialog.Header class="relative flex flex-row items-center justify-between p-2 pr-8">
			<!--
				The arrow is out of the flow and the title slides, so entering a sub-view
				moves the title with a transform instead of reflowing the header. min-h
				keeps the header's height fixed either way.
			-->
			<Button
				aria-label="Back to models"
				class="absolute top-1/2 left-1 h-7 w-7 -translate-y-1/2 transition-[opacity,visibility] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] {view ===
				ModelsDialogView.MANAGE
					? 'invisible opacity-0'
					: 'visible opacity-100'}"
				onclick={() => (view = ModelsDialogView.MANAGE)}
				size="icon"
				variant="ghost"
			>
				<ArrowLeft class="h-4 w-4" />
			</Button>

			<Dialog.Title
				class="flex min-h-7 items-center gap-2 transition-transform duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] {view ===
				ModelsDialogView.MANAGE
					? 'translate-x-0'
					: 'translate-x-8'}"
			>
				<!-- the marks cross-fade in one grid cell, so the title never shifts -->
				<span class="grid h-5 w-5 shrink-0 place-items-center">
					<MODEL_ICON
						class="col-start-1 row-start-1 h-5 w-5 transition-opacity duration-150 {view ===
						ModelsDialogView.MANAGE
							? 'opacity-100'
							: 'opacity-0'}"
					/>

					<DISCOVER_ICON
						class="col-start-1 row-start-1 h-5 w-5 transition-opacity duration-150 {view ===
						ModelsDialogView.DISCOVER
							? 'opacity-100'
							: 'opacity-0'}"
					/>
				</span>

				<span>{title}</span>
			</Dialog.Title>

			<Dialog.Description class="sr-only">
				Browse, load and delete the models the server can serve.
			</Dialog.Description>
		</Dialog.Header>

		<div class="dialog-view min-h-0 flex-1" data-visible={!isSwapping}>
			{#if shownView === ModelsDialogView.MANAGE}
				<ModelsManager class="h-full">
					{#snippet toolbarEnd()}
						{#if discoverEnabled}
							<Button
								class="gap-1.5"
								onclick={() => (view = ModelsDialogView.DISCOVER)}
								size="sm"
								variant="tertiary"
							>
								<DISCOVER_ICON class="h-3.5 w-3.5" />

								Discover Models
							</Button>
						{/if}
					{/snippet}
				</ModelsManager>
			{:else if shownView === ModelsDialogView.DISCOVER}
				<div class="grid h-full overflow-hidden" style="grid-template-columns: auto 1fr;">
					<ModelsDiscover />
				</div>
			{/if}
		</div>
	</Dialog.Content>
</Dialog.Root>

<style>
	/* a view change reads as one surface swapping, not as content teleporting */
	.dialog-view {
		opacity: 0;
		transition: opacity 150ms cubic-bezier(0.23, 1, 0.32, 1);
	}

	.dialog-view[data-visible='true'] {
		opacity: 1;
	}

	@media (prefers-reduced-motion: reduce) {
		.dialog-view {
			transition-duration: 100ms;
		}
	}
</style>
