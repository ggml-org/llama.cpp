<script lang="ts">
	import { replaceState } from '$app/navigation';
	import { page } from '$app/state';
	import { DialogModelNotAvailable } from '$lib/components/app';
	import { APP_NAME, URL_PARAMS } from '$lib/constants';
	import { chatStore, conversationsStore, modelsStore, serverStore } from '$lib/stores';
	import { onMount } from 'svelte';

	let qParam = $derived(page.url.searchParams.get(URL_PARAMS.QUERY));
	let modelParam = $derived(page.url.searchParams.get(URL_PARAMS.MODEL));
	let loadParam = $derived(page.url.searchParams.get(URL_PARAMS.LOAD));

	let showModelNotAvailable = $state(false);
	let requestedModelName = $state('');
	let availableModelNames = $derived(modelsStore.models.map((m) => m.model));

	// Clear params after handling the deep link so a refresh does not replay them
	function clearUrlParams() {
		const url = new URL(page.url);

		url.searchParams.delete(URL_PARAMS.QUERY);
		url.searchParams.delete(URL_PARAMS.MODEL);
		url.searchParams.delete(URL_PARAMS.LOAD);

		replaceState(url.toString(), {});
	}

	async function handleUrlParams() {
		await modelsStore.fetch();

		if (modelParam) {
			const model = modelsStore.findModelByName(modelParam);

			if (model) {
				try {
					await modelsStore.selectModelById(model.id);

					// with ?load=true in router mode, start loading right away so the
					// model is ready sooner; not awaited so the UI stays usable
					if (
						loadParam === 'true' &&
						serverStore.isRouterMode &&
						!modelsStore.isModelLoaded(model.id)
					) {
						modelsStore.status
							.load(model.id)
							.catch((error) => console.error('Failed to load model:', error));
					}
				} catch (error) {
					console.error('Failed to select model:', error);
					requestedModelName = modelParam;
					showModelNotAvailable = true;

					return;
				}
			} else {
				requestedModelName = modelParam;
				showModelNotAvailable = true;

				return;
			}
		}

		// ?q= creates the conversation, the chat route sends the prompt once the
		// conversation id is in the URL
		if (qParam !== null) {
			await conversationsStore.createConversation();
			clearUrlParams();
		} else if (modelParam) {
			clearUrlParams();
		}
	}

	onMount(async () => {
		if (!conversationsStore.isInitialized) {
			await conversationsStore.initialize();
		}

		conversationsStore.clearActiveConversation();
		chatStore.clearUIState();

		await modelsStore.fetch();

		if (qParam !== null || modelParam !== null) {
			await handleUrlParams();
		}

		await modelsStore.ensureFirstModelSelected();
	});
</script>

<svelte:head>
	<title>{APP_NAME}</title>
</svelte:head>

<DialogModelNotAvailable
	bind:open={showModelNotAvailable}
	availableModels={availableModelNames}
	modelName={requestedModelName}
/>
