<script lang="ts">
	import { CircleHelp } from '@lucide/svelte';
	import { ChatMessageActionCard } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import type { AgenticQuestionAnswers, AgenticQuestionPrompt } from '$lib/types';

	interface Props {
		questions: AgenticQuestionPrompt[];
		onAnswer: (answers: AgenticQuestionAnswers) => void;
		onDismiss: () => void;
	}

	let { questions, onAnswer, onDismiss }: Props = $props();
</script>

<ChatMessageActionCard icon={CircleHelp}>
	{#snippet message()}
		{questions[0]?.question ?? 'The assistant has a question.'}
	{/snippet}

	{#snippet actions()}
		<Button variant="secondary" size="sm" onclick={() => onAnswer(questions.map(() => []))}>
			Continue
		</Button>
		<Button variant="destructive" size="sm" onclick={onDismiss}>Dismiss</Button>
	{/snippet}
</ChatMessageActionCard>
