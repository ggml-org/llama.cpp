<script lang="ts">
	import { HF_PREVIEW_LINES_MAX, HF_PREVIEW_TEXT_MAX } from '../constants.js';
	import type { HfTracePreviewLine } from '../types.js';
	import { FileText } from '@lucide/svelte';

	interface Props {
		/** The trace file, as it is named on the Hub. */
		path: string;
		/** The first messages of the conversation; the file alone shows without them. */
		lines?: HfTracePreviewLine[];
	}

	let { lines = [], path }: Props = $props();

	// the first messages only, as much of each as a row can show
	let shown = $derived(lines.slice(0, HF_PREVIEW_LINES_MAX));

	const ROLE_LABELS = { assistant: 'Assistant', user: 'You' } as const;
</script>

<!-- the trace as a file, with its first messages fading out below, as if the rest went on -->
<div
	class="inset-group-surface overflow-hidden motion-safe:animate-in motion-safe:duration-500 motion-safe:ease-out-quint motion-safe:fade-in-0 motion-safe:slide-in-from-bottom-2"
	data-slot="hf-trace-preview"
>
	<div class="flex items-center gap-2 border-b border-(--line-default) px-3 py-2">
		<FileText class="size-icon-md shrink-0 text-muted-foreground" />

		<span class="truncate font-mono text-caption text-muted-foreground">{path}</span>
	</div>

	{#if shown.length}
		<ul class="flex max-h-36 flex-col gap-2.5 overflow-hidden px-3 pt-3 mask-b-from-35%">
			{#each shown as line, index (index)}
				<li class="grid grid-cols-[4.5rem_1fr] items-baseline gap-2">
					<span class="text-caption font-medium text-muted-foreground"
						>{ROLE_LABELS[line.role]}</span
					>

					<span class="line-clamp-2 text-body-sm">{line.text.slice(0, HF_PREVIEW_TEXT_MAX)}</span>
				</li>
			{/each}
		</ul>
	{/if}
</div>
