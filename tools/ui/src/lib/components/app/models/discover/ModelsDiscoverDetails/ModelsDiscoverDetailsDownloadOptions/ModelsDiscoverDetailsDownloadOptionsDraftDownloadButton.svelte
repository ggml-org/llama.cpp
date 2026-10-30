<script lang="ts">
	import { Download } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { HuggingFaceService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import type { ModelCompanionDraft } from '$lib/types';

	interface Props {
		/** The draft its model is missing, with its own file, quant and size. */
		companion: ModelCompanionDraft;
		/** Tag of the model the draft belongs to, the one the server pulls it with. */
		repoWithTag: string;
	}

	let { companion, repoWithTag }: Props = $props();

	// one source, so the chip matches the draft badge of a quant chip
	const SIDECAR_BADGE_CLASS =
		'rounded-md bg-primary px-1 py-0.5 text-[10px] font-semibold tracking-wide text-primary-foreground uppercase';

	let tooltipText = $derived(`Add the ${companion.kind} draft for ${repoWithTag}`);

	// a draft is pulled with its model, so the model is posted again with the kind
	function addDraft() {
		void modelsStore.status.downloadModel(repoWithTag, [companion.kind]).catch(() => {
			// the store already toasted the failure
		});
	}
</script>

<!-- A downloaded model missing its draft: the chip adds the draft alone, and
     the model keeps the files it already has. -->
<Tooltip.Root>
	<Tooltip.Trigger>
		{#snippet child({ props })}
			<button
				{...props}
				aria-label={tooltipText}
				class="group relative inline-flex h-auto cursor-pointer items-center gap-1 rounded-md border border-dashed border-border/60 px-2 py-1 text-left font-mono text-xs shadow-sm transition-[background-color,border-color,transform] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] hover:bg-muted-foreground/10 hover:border-border active:scale-[0.97] dark:border-border/40 dark:hover:bg-muted-foreground/15"
				onclick={addDraft}
				type="button"
			>
				<span class="{SIDECAR_BADGE_CLASS} inline-flex items-center gap-0.5">
					{companion.kind}{companion.quant ? `:${companion.quant}` : ''}
				</span>

				<span class="-my-1 mx-0.75 w-px self-stretch bg-border"></span>

				<span>{HuggingFaceService.formatFileSize(companion.size)}</span>

				<Download
					class="size-icon-sm text-muted-foreground transition-colors duration-150 group-hover:text-foreground"
				/>
			</button>
		{/snippet}
	</Tooltip.Trigger>

	<Tooltip.Content>
		<p>{tooltipText}</p>
	</Tooltip.Content>
</Tooltip.Root>
