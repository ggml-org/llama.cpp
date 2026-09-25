<script lang="ts">
	import ChatFormPromptPickerArgumentInput from './ChatFormPromptPickerArgumentInput.svelte';
	import { Button } from '$lib/components/ui/button';
	import type { MCPPromptInfo } from '$lib/types';

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
		activeAutocomplete,
		autocompleteIndex,
		loadingSuggestions,
		onArgBlur,
		onArgFocus,
		onArgInput,
		onArgKeydown,
		onCancel,
		onSelectSuggestion,
		onSubmit,
		prompt,
		promptArgs,
		promptError,
		suggestions
	}: Props = $props();
</script>

<form class="space-y-3 pt-4" onsubmit={onSubmit}>
	{#each prompt.arguments ?? [] as arg (arg.name)}
		<ChatFormPromptPickerArgumentInput
			argument={arg}
			autocompleteIndex={activeAutocomplete === arg.name ? autocompleteIndex : 0}
			isAutocompleteActive={activeAutocomplete === arg.name}
			isLoadingSuggestions={loadingSuggestions[arg.name] ?? false}
			onBlur={() => onArgBlur(arg.name)}
			onFocus={() => onArgFocus(arg.name)}
			onInput={(value) => onArgInput(arg.name, value)}
			onKeydown={(e) => onArgKeydown(e, arg.name)}
			onSelectSuggestion={(value) => onSelectSuggestion(arg.name, value)}
			suggestions={suggestions[arg.name] ?? []}
			value={promptArgs[arg.name] ?? ''}
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

	<div class="mt-8 flex justify-end gap-2">
		<Button onclick={onCancel} size="sm" type="button" variant="secondary">Cancel</Button>

		<Button size="sm" type="submit">Use Prompt</Button>
	</div>
</form>
