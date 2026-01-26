<script lang="ts">
	import type { MCPPromptInfo } from '$lib/types';
	import ChatFormPromptPickerArgumentInput from './ChatFormPromptPickerArgumentInput.svelte';

	interface Props {
		prompt: MCPPromptInfo;
		promptArgs: Record<string, string>;
		suggestions: Record<string, string[]>;
		loadingSuggestions: Record<string, boolean>;
		activeAutocomplete: string | null;
		autocompleteIndex: number;
		promptError: string | null;
		onArgInput: (argName: string, value: string) => void;
		onArgKeydown: (event: KeyboardEvent, argName: string) => void;
		onArgBlur: (argName: string) => void;
		onArgFocus: (argName: string) => void;
		onSelectSuggestion: (argName: string, value: string) => void;
		onSubmit: (event: SubmitEvent) => void;
		onCancel: () => void;
	}

	let {
		prompt,
		promptArgs,
		suggestions,
		loadingSuggestions,
		activeAutocomplete,
		autocompleteIndex,
		promptError,
		onArgInput,
		onArgKeydown,
		onArgBlur,
		onArgFocus,
		onSelectSuggestion,
		onSubmit,
		onCancel
	}: Props = $props();
</script>

<form onsubmit={onSubmit} class="space-y-3 pt-4">
	{#each prompt.arguments ?? [] as arg (arg.name)}
		<ChatFormPromptPickerArgumentInput
			argument={arg}
			value={promptArgs[arg.name] ?? ''}
			suggestions={suggestions[arg.name] ?? []}
			isLoadingSuggestions={loadingSuggestions[arg.name] ?? false}
			isAutocompleteActive={activeAutocomplete === arg.name}
			autocompleteIndex={activeAutocomplete === arg.name ? autocompleteIndex : 0}
			onInput={(value) => onArgInput(arg.name, value)}
			onKeydown={(e) => onArgKeydown(e, arg.name)}
			onBlur={() => onArgBlur(arg.name)}
			onFocus={() => onArgFocus(arg.name)}
			onSelectSuggestion={(value) => onSelectSuggestion(arg.name, value)}
		/>
	{/each}

	{#if promptError}
		<div
			class="flex items-start gap-2 rounded-lg border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive"
			role="alert"
		>
			<span class="shrink-0">⚠</span>
			<span>{promptError}</span>
		</div>
	{/if}

	<div class="flex justify-end gap-2">
		<button
			type="button"
			onclick={onCancel}
			class="rounded-lg px-3 py-1.5 text-sm text-muted-foreground hover:bg-accent"
		>
			Cancel
		</button>

		<button
			type="submit"
			class="rounded-lg bg-primary px-3 py-1.5 text-sm text-primary-foreground hover:bg-primary/90"
		>
			Use Prompt
		</button>
	</div>
</form>
