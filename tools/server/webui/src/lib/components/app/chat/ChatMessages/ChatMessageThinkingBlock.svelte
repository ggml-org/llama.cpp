<script lang="ts">
	import { Brain } from '@lucide/svelte';
	import { config } from '$lib/stores/settings.svelte';
	import CollapsibleInfoCard from './CollapsibleInfoCard.svelte';

	interface Props {
		class?: string;
		hasRegularContent?: boolean;
		isStreaming?: boolean;
		reasoningContent: string | null;
	}

	let {
		class: className = '',
		hasRegularContent = false,
		isStreaming = false,
		reasoningContent
	}: Props = $props();

	const currentConfig = config();

	let isExpanded = $state(currentConfig.showThoughtInProgress);

	$effect(() => {
		if (hasRegularContent && reasoningContent && currentConfig.showThoughtInProgress) {
			isExpanded = false;
		}
	});
</script>

<CollapsibleInfoCard
	bind:open={isExpanded}
	class="mb-6 {className}"
	icon={Brain}
	title={isStreaming ? 'Reasoning...' : 'Reasoning'}
>
	<div class="pt-3">
		<div class="text-xs leading-relaxed break-words whitespace-pre-wrap">
			{reasoningContent ?? ''}
		</div>
	</div>
</CollapsibleInfoCard>
