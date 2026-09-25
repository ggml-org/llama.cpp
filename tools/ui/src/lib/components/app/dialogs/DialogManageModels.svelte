<script lang="ts">
	import { ArrowLeft, Box, Compass, Server } from '@lucide/svelte';
	import ModelsDiscover from '$lib/components/app/models/discover/ModelsDiscover.svelte';
	import ModelsManager from '$lib/components/app/models/ModelsManager/ModelsManager.svelte';
	import ModelsManagerModelProviders from '$lib/components/app/models/ModelsManager/ModelsManagerModelProviders.svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { uiStore } from '$lib/stores';
	import { untrack } from 'svelte';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { onOpenChange, open = $bindable(false) }: Props = $props();

	type View = 'discover' | 'manage' | 'providers';

	/** How long a view takes to fade out before the next one fades in. */
	const VIEW_FADE_MS = 150;

	// what the user asked for, and what is actually rendered: a view change fades the
	// current one out, swaps, then fades the next one in
	let view = $state<View>('manage');
	let shownView = $state<View>('manage');
	let isSwapping = $state(false);

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
		}, VIEW_FADE_MS);

		return () => clearTimeout(timer);
	});

	let title = $derived(
		view === 'discover' ? 'Discover' : view === 'providers' ? 'Providers' : 'Models'
	);

	// the sidebar's Discover entry opens this dialog on its Discover view
	$effect(() => {
		if (!uiStore.discoverModelsOpen) return;

		uiStore.discoverModelsOpen = false;
		view = 'discover';
		handleOpenChange(true);
	});

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}
</script>

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content
		class="md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-380! flex flex-col p-4 pb-0"
		onCloseAutoFocus={(event) => event.preventDefault()}
		onOpenAutoFocus={(event) => event.preventDefault()}
	>
		<Dialog.Header class="flex flex-row items-center justify-between p-2 pr-8">
			<!-- min-h keeps the back button from growing the header and shifting the body -->
			<Dialog.Title class="flex min-h-7 items-center gap-2">
				{#if view !== 'manage'}
					<Button
						aria-label="Back to models"
						class="-ml-1 h-7 w-7"
						onclick={() => (view = 'manage')}
						size="icon"
						variant="ghost"
					>
						<ArrowLeft class="h-4 w-4" />
					</Button>
				{/if}

				{#if view === 'manage'}
					<Box class="h-5 w-5" />
				{:else if view === 'discover'}
					<Compass class="h-5 w-5" />
				{:else}
					<Server class="h-5 w-5" />
				{/if}

				<span>{title}</span>
			</Dialog.Title>
		</Dialog.Header>

		<div class="dialog-view min-h-0 flex-1" data-visible={!isSwapping}>
			{#if shownView === 'manage'}
				<ModelsManager class="h-full">
					{#snippet toolbarEnd()}
						<Button
							class="gap-1.5"
							onclick={() => (view = 'discover')}
							size="sm"
							variant="secondary"
						>
							<Compass class="h-3.5 w-3.5" />

							Discover Models
						</Button>

						<Button
							class="gap-1.5"
							onclick={() => (view = 'providers')}
							size="sm"
							variant="secondary"
						>
							<Server class="h-3.5 w-3.5" />

							Manage Providers
						</Button>
					{/snippet}
				</ModelsManager>
			{:else if shownView === 'discover'}
				<div class="grid h-full overflow-hidden" style="grid-template-columns: auto 1fr;">
					<ModelsDiscover />
				</div>
			{:else}
				<div class="h-full overflow-y-auto">
					<ModelsManagerModelProviders />
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
