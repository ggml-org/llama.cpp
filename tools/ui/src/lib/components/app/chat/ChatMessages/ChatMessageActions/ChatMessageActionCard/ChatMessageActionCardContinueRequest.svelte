<script lang="ts">
	import { RotateCw } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import ChatMessageActionCard from './ChatMessageActionCard.svelte';

	interface Props {
		onDecision: (shouldContinue: boolean) => void;
	}

	let { onDecision }: Props = $props();
</script>

<ChatMessageActionCard icon={RotateCw}>
	{#snippet message()}
		Agentic turn limit reached. Continue?
	{/snippet}

	{#snippet actions()}
		<Button size="sm" onclick={() => onDecision(true)}>Continue</Button>

		<Button
			variant="destructive"
			size="sm"
			class="text-destructive hover:text-destructive"
			onclick={() => onDecision(false)}
		>
			Stop
		</Button>
	{/snippet}
</ChatMessageActionCard>
