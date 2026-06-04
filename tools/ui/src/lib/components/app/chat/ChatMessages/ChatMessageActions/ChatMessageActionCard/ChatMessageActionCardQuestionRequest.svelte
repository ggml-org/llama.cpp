<script lang="ts">
	import { MessageSquare } from '@lucide/svelte';
	import CheckIcon from '@lucide/svelte/icons/check';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import type { PendingBuiltinQuestionItem } from '$lib/types/agentic';
	import ChatMessageActionCard from './ChatMessageActionCard.svelte';

	interface Props {
		questions: PendingBuiltinQuestionItem[];
		onAnswer: (answers: string[][]) => void;
		onDismiss?: () => void;
	}

	let { questions, onAnswer, onDismiss }: Props = $props();

	let currentIndex = $state(0);
	let selections = $state<Record<number, string[]>>({});
	let customAnswers = $state<Record<number, string>>({});
	let customSelections = $state<Record<number, boolean>>({});

	$effect(() => {
		if (questions.length === 0) {
			currentIndex = 0;
			return;
		}

		if (currentIndex >= questions.length) {
			currentIndex = questions.length - 1;
		}
	});

	const currentQuestion = $derived(questions[currentIndex] ?? null);
	const showProgress = $derived(questions.length > 1);
	const isLastQuestion = $derived(currentIndex >= questions.length - 1);
	const isSingleQuestion = $derived(questions.length === 1);
	const isDirectSingleChoice = $derived(
		isSingleQuestion && questionType(currentQuestion) === 'single_choice'
	);
	const primaryActionDisabled = $derived.by(() => {
		if (!currentQuestion) return true;
		return !answered(currentIndex);
	});
	const progressLabel = $derived(
		questions.length === 1
			? '1 of 1 question'
			: `${Math.min(currentIndex + 1, questions.length)} of ${questions.length} questions`
	);

	function questionType(question: PendingBuiltinQuestionItem | null): 'freeform' | 'multiple_choice' | 'single_choice' {
		if (!question) return 'single_choice';
		if (!question.options || question.options.length === 0) return 'freeform';
		return question.multiple ? 'multiple_choice' : 'single_choice';
	}

	function questionHint(question: PendingBuiltinQuestionItem | null): string {
		switch (questionType(question)) {
			case 'multiple_choice':
				return 'Select all answers that apply';
			case 'freeform':
				return 'Type your own answer';
			default:
				return 'Select one answer';
		}
	}

	function answered(questionIndex: number): boolean {
		const question = questions[questionIndex];
		if (!question) return false;

		const optionAnswers = selections[questionIndex] ?? [];
		const customAnswer = customAnswers[questionIndex]?.trim();
		const type = questionType(question);

		if (type === 'freeform') {
			return Boolean(customAnswer);
		}

		if (type === 'multiple_choice') {
			return optionAnswers.length > 0 || (isCustomSelected(questionIndex, question) && Boolean(customAnswer));
		}

		return optionAnswers.length > 0 || (isCustomSelected(questionIndex, question) && Boolean(customAnswer));
	}

	function isCustomSelected(questionIndex: number, question: PendingBuiltinQuestionItem): boolean {
		if (questionType(question) === 'freeform') return true;
		return customSelections[questionIndex] === true;
	}

	function toggleOption(questionIndex: number, label: string, question: PendingBuiltinQuestionItem) {
		const current = selections[questionIndex] ?? [];
		const type = questionType(question);

		if (type !== 'multiple_choice') {
			selections[questionIndex] = current.includes(label) ? [] : [label];
			customSelections[questionIndex] = false;
			return;
		}

		selections[questionIndex] = current.includes(label)
			? current.filter((item) => item !== label)
			: [...current, label];
	}

	function isSelected(questionIndex: number, label: string): boolean {
		return selections[questionIndex]?.includes(label) ?? false;
	}

	function selectOption(questionIndex: number, label: string, question: PendingBuiltinQuestionItem) {
		if (isDirectSingleChoice && questionType(question) === 'single_choice') {
			onAnswer([[label]]);
			return;
		}

		toggleOption(questionIndex, label, question);
	}

	function selectCustom(questionIndex: number, question: PendingBuiltinQuestionItem) {
		const type = questionType(question);
		if (type === 'freeform') return;

		if (type === 'multiple_choice') {
			customSelections[questionIndex] = !isCustomSelected(questionIndex, question);
			return;
		}

		customSelections[questionIndex] = true;
		selections[questionIndex] = [];
	}

	function jumpToQuestion(index: number) {
		currentIndex = index;
	}

	function goBack() {
		if (currentIndex <= 0) return;
		currentIndex -= 1;
	}

	function goForward() {
		if (primaryActionDisabled) return;

		if (isLastQuestion) {
			submit();
			return;
		}

		currentIndex += 1;
	}

	function submit() {
		const answers = questions.map((question, index) => {
			const optionAnswers = selections[index] ?? [];
			const customAnswer = customAnswers[index]?.trim();
			const type = questionType(question);

			if (type === 'freeform') {
				return customAnswer ? [customAnswer] : [];
			}

			if (type === 'multiple_choice') {
				return isCustomSelected(index, question) && customAnswer
					? [...optionAnswers, customAnswer]
					: optionAnswers;
			}

			return isCustomSelected(index, question) && customAnswer ? [customAnswer] : optionAnswers;
		});

		onAnswer(answers);
	}

	function handleInputKeydown(event: KeyboardEvent) {
		if (event.key !== 'Enter') return;
		if (event.shiftKey) return;
		if ((event.metaKey || event.ctrlKey) && !event.altKey) return;

		event.preventDefault();
		goForward();
	}
</script>

<ChatMessageActionCard icon={MessageSquare}>
	{#snippet message()}
		The agent has a question before continuing.
	{/snippet}

	{#snippet actions()}
		<div class="question-request flex w-full flex-col gap-4">
			{#if showProgress}
				<div class="question-header">
					<div class="question-header-title">{progressLabel}</div>
					<div class="question-progress">
						{#each questions as _, questionIndex (questionIndex)}
							<button
								type="button"
								class="question-progress-segment"
								data-active={questionIndex === currentIndex}
								data-answered={answered(questionIndex)}
								onclick={() => jumpToQuestion(questionIndex)}
								aria-label={`Question ${questionIndex + 1}`}
							></button>
						{/each}
					</div>
				</div>
			{/if}

			{#if currentQuestion}
				<div class="question-content">
					<div class="question-text">{currentQuestion.question}</div>
					<div class="question-hint">{questionHint(currentQuestion)}</div>

					<div class="question-options" role="group" aria-label={currentQuestion.header}>
						{#each currentQuestion.options ?? [] as option (option.label)}
							<button
								type="button"
							data-slot="question-option"
							data-picked={isSelected(currentIndex, option.label)}
							role={questionType(currentQuestion) === 'multiple_choice' ? 'checkbox' : 'radio'}
							aria-checked={isSelected(currentIndex, option.label)}
							onclick={() => selectOption(currentIndex, option.label, currentQuestion)}
						>
							{#if !isDirectSingleChoice}
								<span data-slot="question-option-check">
									<span
										data-slot="question-option-box"
										data-type={questionType(currentQuestion) === 'multiple_choice' ? 'checkbox' : 'radio'}
										data-picked={isSelected(currentIndex, option.label)}
									>
										<CheckIcon class="h-3 w-3" />
									</span>
								</span>
							{/if}

								<span data-slot="question-option-main">
									<span data-slot="option-label">{option.label}</span>
									{#if option.description}
										<span data-slot="option-description">{option.description}</span>
									{/if}
								</span>
							</button>
						{/each}

						{#if questionType(currentQuestion) === 'freeform'}
							<Input
								bind:value={customAnswers[currentIndex]}
								placeholder="Type your answer..."
								class="h-9 text-sm"
								onkeydown={handleInputKeydown}
							/>
						{:else if currentQuestion.custom !== false}
							<button
								type="button"
								data-slot="question-option"
								data-picked={isCustomSelected(currentIndex, currentQuestion)}
								data-custom="true"
								role={questionType(currentQuestion) === 'multiple_choice' ? 'checkbox' : 'radio'}
								aria-checked={isCustomSelected(currentIndex, currentQuestion)}
								onclick={() => selectCustom(currentIndex, currentQuestion)}
							>
								{#if !isDirectSingleChoice}
									<span data-slot="question-option-check">
										<span
											data-slot="question-option-box"
											data-type={questionType(currentQuestion) === 'multiple_choice' ? 'checkbox' : 'radio'}
											data-picked={isCustomSelected(currentIndex, currentQuestion)}
										>
											<CheckIcon class="h-3 w-3" />
										</span>
									</span>
								{/if}

								<span data-slot="question-option-main">
									<span data-slot="option-label">Type your own answer</span>
									<span data-slot="option-description">
										{customAnswers[currentIndex]?.trim() || 'Type your answer...'}
									</span>
								</span>
							</button>

							{#if isCustomSelected(currentIndex, currentQuestion)}
								<Input
									bind:value={customAnswers[currentIndex]}
									placeholder="Type your answer..."
									class="h-8 text-sm"
									onkeydown={handleInputKeydown}
								/>
							{/if}
						{/if}
					</div>
				</div>
			{/if}

			<div class="question-footer">
				{#if onDismiss}
					<Button
						variant="destructive"
						size="sm"
						class="text-destructive hover:text-destructive"
						onclick={onDismiss}
					>
						Dismiss
					</Button>
				{/if}

				{#if !isDirectSingleChoice}
					<div class="question-footer-actions">
						{#if currentIndex > 0}
							<Button variant="secondary" size="sm" onclick={goBack}>Back</Button>
						{/if}

						<Button
							variant={isLastQuestion ? 'default' : 'secondary'}
							size="sm"
							disabled={primaryActionDisabled}
							onclick={goForward}
						>
							{isLastQuestion ? 'Submit' : 'Next'}
						</Button>
					</div>
				{/if}
			</div>
		</div>
	{/snippet}
</ChatMessageActionCard>

<style>
	.question-request {
		min-width: 0;
	}

	.question-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.75rem;
	}

	.question-header-title {
		font-size: 0.875rem;
		font-weight: 600;
		color: hsl(var(--foreground));
	}

	.question-progress {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		flex-shrink: 0;
	}

	.question-progress-segment {
		width: 1rem;
		height: 1rem;
		padding: 0;
		border: 0;
		border-radius: 9999px;
		background: transparent;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		cursor: pointer;
	}

	.question-progress-segment::after {
		content: '';
		width: 1rem;
		height: 2px;
		border-radius: 9999px;
		background: hsl(var(--muted-foreground) / 0.35);
		transition: background-color 120ms ease;
	}

	.question-progress-segment[data-active='true']::after {
		background: hsl(var(--foreground));
	}

	.question-progress-segment[data-answered='true']::after {
		background: hsl(var(--primary));
	}

	.question-content {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.question-text {
		font-size: 0.9375rem;
		font-weight: 600;
		line-height: 1.45;
		color: hsl(var(--foreground));
	}

	.question-hint {
		font-size: 0.8125rem;
		color: hsl(var(--muted-foreground));
	}

	.question-options {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		margin-top: 0.75rem;
	}

	[data-slot='question-option'] {
		width: 100%;
		display: flex;
		align-items: flex-start;
		gap: 0.75rem;
		padding: 0.75rem 0.875rem;
		border: 0;
		border-radius: 0.5rem;
		background: color-mix(in oklch, var(--muted) 98%, var(--foreground) 2%);
		text-align: left;
		cursor: pointer;
		box-shadow: 0 1px 2px color-mix(in oklch, var(--foreground) 5%, transparent);
		transition:
			background 120ms ease,
			box-shadow 120ms ease;
	}

	[data-slot='question-option']:hover:not([data-picked='true']) {
		background: color-mix(in oklch, var(--muted) 96%, var(--foreground) 4%);
	}

	[data-slot='question-option'][data-picked='true'] {
		background: color-mix(in oklch, var(--primary) 12%, var(--background) 88%);
		box-shadow: 0 1px 2px color-mix(in oklch, var(--primary) 9%, transparent);
	}

	[data-slot='question-option']:focus-visible {
		outline: none;
		box-shadow: 0 0 0 2px hsl(var(--ring) / 0.2);
	}

	[data-slot='question-option-check'] {
		display: inline-flex;
		transform: translateY(2px);
	}

	[data-slot='question-option-box'] {
		width: 1rem;
		height: 1rem;
		padding: 2px;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
		border-radius: 0.25rem;
		background: hsl(var(--background) / 0.75);
		color: hsl(var(--primary-foreground));
		transition:
			background 120ms ease,
			border-color 120ms ease;
	}

	[data-slot='question-option-box'] :global(svg) {
		opacity: 0;
		transition: opacity 120ms ease;
	}

	[data-slot='question-option-box'][data-type='radio'] {
		border-radius: 9999px;
	}

	[data-slot='question-option-box'][data-picked='true'] {
		background: hsl(var(--primary));
	}

	[data-slot='question-option-box'][data-picked='true'] :global(svg) {
		opacity: 1;
	}

	[data-slot='question-option-main'] {
		display: flex;
		flex-direction: column;
		gap: 0.125rem;
		min-width: 0;
		flex: 1;
	}

	[data-slot='option-label'] {
		font-size: 0.875rem;
		font-weight: 600;
		line-height: 1.35;
		color: hsl(var(--foreground));
	}

	[data-slot='option-description'] {
		font-size: 0.8125rem;
		line-height: 1.35;
		color: hsl(var(--muted-foreground));
		white-space: normal;
		overflow-wrap: anywhere;
	}

	.question-footer {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.75rem;
	}

	.question-footer-actions {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-left: auto;
	}
</style>
