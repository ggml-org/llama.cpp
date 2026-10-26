<script lang="ts">
	import ModelsManagerModelConfiguration from './ModelsManagerModelConfiguration/ModelsManagerModelConfiguration.svelte';
	import { downloadGroups, findModelOption } from './utils';
	import * as Drawer from '$lib/components/ui/drawer';
	import { conversationsStore, deviceStore, modelsStore, serverStore, uiStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { untrack } from 'svelte';

	// tracked downloads are not in the listing yet
	let downloads = $derived(
		downloadGroups(modelsStore.status.getDownloadEntries(), modelsStore.models, (repo) =>
			modelsStore.status.getDraftSidecars(repo)
		)
	);

	let selected = $derived(
		findModelOption(uiStore.modelDetailsFocus, modelsStore.models, downloads)
	);
	// keep the last model on screen while the drawer closes
	let lastPicked = $state<ModelOption | null>(null);
	let target = $derived(selected ?? lastPicked);
	let shownId = $state<string | null>(null);
	let isSwapping = $state(false);
	let fade = $state<'open' | 'swap'>('open');
	let shownOption = $derived(findModelOption(shownId, modelsStore.models, downloads));

	$effect(() => {
		const option = selected;

		if (!option) return;

		// untracked: the effect must not track the state it writes
		untrack(() => {
			lastPicked = option;
		});
	});

	/** How long the panel takes to fade out before it swaps to another model. */
	const SWAP_FADE_MS = 120;

	// Another model fades the panel out, swaps it, then fades it back in.
	$effect(() => {
		const next = target?.id ?? null;

		if (!next) return;

		if (shownId === null) {
			untrack(() => (shownId = next));

			return;
		}

		if (next === shownId) {
			if (selected) {
				untrack(() => {
					isSwapping = false;
					fade = 'open';
				});
			}

			return;
		}

		untrack(() => {
			isSwapping = true;
			fade = 'swap';
		});

		const timer = setTimeout(() => {
			untrack(() => {
				shownId = next;
				isSwapping = false;
			});
		}, SWAP_FADE_MS);

		return () => clearTimeout(timer);
	});

	async function toggleLoad(option: ModelOption): Promise<void> {
		if (modelsStore.isModelLoaded(option.model)) {
			await modelsStore.status.unload(option.model);

			return;
		}

		await modelsStore.status.load(option.model);
	}

	/** Close the drawer, the picker and the models page, then focus the composer. */
	function returnToChat(): void {
		uiStore.closeModelDetails();
		uiStore.closeModelsSelector();
		uiStore.requestComposerFocus();
		void uiStore.closeModelsManager();
	}

	/** Load the model when the server can take load requests, without waiting for it. */
	function loadInBackground(option: ModelOption): void {
		// only the built-in server loads on request, and only in router mode
		if (serverStore.isRouterMode && !modelsStore.isModelLoaded(option.model)) {
			modelsStore.status
				.load(option.model)
				.catch((error) => console.error('Failed to load model:', error));
		}
	}

	/** Switch the open chat to this model, the way the desktop model dropdown does. */
	async function useInChat(option: ModelOption): Promise<void> {
		// a deliberate pick records the recency; useInNewChat skips it because a new
		// conversation is where the pick is exercised, not a return to a model the
		// user went looking for
		await modelsStore.selectModelById(option.id, { recordRecent: true });

		loadInBackground(option);

		returnToChat();
	}

	async function useInNewChat(option: ModelOption): Promise<void> {
		await modelsStore.selectModelById(option.id);
		await conversationsStore.openNewChat();

		loadInBackground(option);

		returnToChat();
	}
</script>

<!-- bound both ways: vaul writes its own open state on a drag to dismiss, and a one-way prop would then keep it closed -->
<Drawer.Root
	bind:open={
		() => selected !== null,
		(open) => {
			if (!open) uiStore.closeModelDetails();
		}
	}
	direction={deviceStore.isMobile ? 'bottom' : 'right'}
	shouldScaleBackground={false}
>
	<Drawer.Content
		class="data-[vaul-drawer-direction=bottom]:h-[85dvh] data-[vaul-drawer-direction=bottom]:max-h-[85dvh] data-[vaul-drawer-direction=right]:w-[30rem] data-[vaul-drawer-direction=right]:max-w-[calc(100vw-2rem)] data-[vaul-drawer-direction=right]:sm:max-w-[30rem]"
	>
		<Drawer.Title class="sr-only">Model details</Drawer.Title>

		<Drawer.Description class="sr-only">
			Configuration, actions and metadata of the selected model.
		</Drawer.Description>

		<div
			class="pane-content flex min-h-0 flex-1 flex-col"
			data-fade={fade}
			data-visible={selected !== null && !isSwapping}
		>
			{#if shownOption}
				{#key shownId}
					<ModelsManagerModelConfiguration
						onClose={() => uiStore.closeModelDetails()}
						onToggleLoad={() => void toggleLoad(shownOption)}
						onUseInChat={() => void useInChat(shownOption)}
						onUseInNewChat={() => void useInNewChat(shownOption)}
						option={shownOption}
					/>
				{/key}
			{/if}
		</div>
	</Drawer.Content>
</Drawer.Root>

<style>
	.pane-content {
		opacity: 0;
		transition: opacity 120ms cubic-bezier(0.23, 1, 0.32, 1);
	}

	.pane-content[data-visible='true'] {
		opacity: 1;
	}

	/* opening: the fade waits for the drawer to move */
	.pane-content[data-visible='true'][data-fade='open'] {
		transition: opacity 150ms cubic-bezier(0.23, 1, 0.32, 1) 80ms;
	}

	/* swapping models: out, then in, with no pause */
	.pane-content[data-visible='true'][data-fade='swap'] {
		transition: opacity 120ms cubic-bezier(0.23, 1, 0.32, 1);
	}
</style>
