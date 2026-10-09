<script lang="ts">
	import { downloadProgressFor } from './utils';
	import { ActionIcon } from '$lib/components/app';
	import { MODEL_DOWNLOAD_ICONS } from '$lib/constants';
	import { ModelRowDownloadState } from '$lib/enums';
	import { modelsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		class?: string;
		option: ModelOption;
		/** Download state the row stands for. */
		state: ModelRowDownloadState;
	}

	let { class: className = '', option, state }: Props = $props();

	let isPaused = $derived(state === ModelRowDownloadState.PAUSED);
	let progress = $derived(downloadProgressFor(option.model));
	let percent = $derived(
		progress && progress.totalBytes > 0
			? Math.round((progress.downloadedBytes / progress.totalBytes) * 100)
			: null
	);
</script>

<!-- the percent is what the row shows at rest; the action takes its place on hover,
     the way the load control does in the other sections -->
<div class={['flex items-center justify-center', className]}>
	<span
		class="text-xs text-muted-foreground tabular-nums group-hover:hidden [@media(pointer:coarse)]:hidden"
	>
		{percent !== null ? `${percent}%` : isPaused ? 'Paused' : ''}
	</span>

	<div class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
		<ActionIcon
			class="h-5 w-5 hover:text-foreground"
			icon={isPaused ? MODEL_DOWNLOAD_ICONS.resume : MODEL_DOWNLOAD_ICONS.pause}
			iconSize="h-4 w-4"
			onclick={() =>
				void (isPaused
					? modelsStore.status.downloadModel(option.model)
					: modelsStore.status.pauseDownload(option.model))}
			stopPropagationOnClick
			tooltip={isPaused ? 'Resume downloading' : 'Pause downloading'}
			tooltipAsTitle
		/>
	</div>
</div>
