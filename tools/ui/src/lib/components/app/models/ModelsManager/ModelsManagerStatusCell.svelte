<script lang="ts">
	import ModelLoadControl from '../ModelLoadControl.svelte';
	import { ServerModelStatus } from '$lib/enums';
	import { modelsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		class?: string;
		option: ModelOption;
	}

	let { class: className = '', option }: Props = $props();

	let status = $derived(modelsStore.getModelStatus(option.model));
	let isOperationInProgress = $derived(modelsStore.status.isOperationInProgress(option.model));
	let isLoaded = $derived(
		(status === ServerModelStatus.LOADED || status === ServerModelStatus.SLEEPING) &&
			!isOperationInProgress
	);
</script>

<ModelLoadControl
	class="justify-self-center {className}"
	isFailed={status === ServerModelStatus.FAILED}
	{isLoaded}
	isLoading={status === ServerModelStatus.LOADING || isOperationInProgress}
	isSleeping={status === ServerModelStatus.SLEEPING}
	{option}
/>
