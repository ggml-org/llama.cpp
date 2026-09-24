<script lang="ts">
	import ModelAvatar from './ModelAvatar.svelte';
	import type { ModelDraft } from './ModelsManager/utils';
	import { Zap } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import type { ModelOption } from '$lib/types/models';
	import { rawModelId } from '$lib/utils/model-option-id';

	interface Props {
		/** The draft model's option, when the manager lists it, so its avatar can be shown. */
		option?: ModelOption | null;
		draft: ModelDraft;
	}

	let { draft, option = null }: Props = $props();
</script>

{#if draft.active}
	<Badge
		class="h-5 max-w-40 gap-1 px-1.5 text-[10px]"
		title="Speculative draft in use"
		variant="secondary"
	>
		<Zap class="h-3 w-3 shrink-0" />

		{#if option}
			<ModelAvatar {option} showRepoOrgAvatar size="size-3.5" />
		{/if}

		<span class="truncate">{rawModelId(option?.model ?? draft.model ?? draft.kind ?? 'draft')}</span
		>
	</Badge>
{:else}
	<Badge
		class="h-5 gap-1 px-1.5 text-[10px] text-muted-foreground/70"
		title="Draft sidecar in this repo, not in use"
		variant="outline"
	>
		{draft.kind}
	</Badge>
{/if}
