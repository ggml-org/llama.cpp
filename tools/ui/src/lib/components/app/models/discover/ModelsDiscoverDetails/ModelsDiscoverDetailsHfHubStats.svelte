<script lang="ts">
	import { Download, Heart } from '@lucide/svelte';
	import { HuggingFaceService } from '$lib/services';
	import type { HfModelDetailInfo } from '$lib/types/huggingface';

	interface Props {
		details: HfModelDetailInfo;
	}

	let { details }: Props = $props();
</script>

<div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-muted-foreground">
	{#if typeof details.downloads === 'number'}
		<span class="inline-flex items-center gap-1.5">
			<Download class="size-icon-sm" />
			{HuggingFaceService.formatDownloads(details.downloads)}
		</span>
	{/if}

	{#if typeof details.likes === 'number'}
		<span class="inline-flex items-center gap-1.5">
			<Heart class="size-icon-sm" />
			{HuggingFaceService.formatLikes(details.likes)}
		</span>
	{/if}

	{#if details.lastModified}
		<span>Updated {HuggingFaceService.formatRelativeTime(details.lastModified)}</span>
	{/if}
</div>
