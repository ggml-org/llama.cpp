<script lang="ts">
	import ModelLoadControl from '../ModelLoadControl.svelte';
	import ModelsManagerDownloadControl from './ModelsManagerDownloadControl.svelte';
	import type { ModelRowDownloadState } from '#lib/enums/index.ts';
	import { ServerModelStatus } from '#lib/enums/index.ts';
	import { modelsStore, serverStore } from '#lib/stores/index.ts';
	import type { ModelOption } from '#lib/types/models.js';

	interface Props {
		class?: string;
		/** Download state, when the row stands for a tracked download. */
		download?: ModelRowDownloadState | null;
		option: ModelOption;
	}

	let { class: className = '', download = null, option }: Props = $props();

	let status = $derived(modelsStore.getModelStatus(option.model));
	let isOperationInProgress = $derived(modelsStore.status.isOperationInProgress(option.model));
	let isLoaded = $derived(modelsStore.isModelRunning(option.model));
	// a MODEL server serves the model it was started with and answers no load route
	let isRouter = $derived(serverStore.isRouterMode);
</script>

{#if download}
	<ModelsManagerDownloadControl class="justify-self-center {className}" {option} state={download} />
{:else if isRouter}
	<ModelLoadControl
		class="justify-self-center {className}"
		isFailed={status === ServerModelStatus.FAILED}
		{isLoaded}
		isLoading={status === ServerModelStatus.LOADING || isOperationInProgress}
		isSleeping={status === ServerModelStatus.SLEEPING}
		{option}
	/>
{:else}
	<span
		class="justify-self-center text-body-sm text-muted-foreground {className}"
		title="Served by this server"
	>
		-
	</span>
{/if}
