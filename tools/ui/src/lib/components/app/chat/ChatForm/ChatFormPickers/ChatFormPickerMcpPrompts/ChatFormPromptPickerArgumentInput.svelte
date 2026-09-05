<script lang="ts">
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import type { MCPPromptInfo } from '$lib/types';
	import { fly } from 'svelte/transition';

	type PromptArgument = NonNullable<MCPPromptInfo['arguments']>[number];

	interface Props {
		argument: PromptArgument;
		value: string;
		suggestions?: string[];
		isLoadingSuggestions?: boolean;
		isAutocompleteActive?: boolean;
		autocompleteIndex?: number;
		onInput: (value: string) => void;
		onKeydown: (event: KeyboardEvent) => void;
		onBlur: () => void;
		onFocus: () => void;
		onSelectSuggestion: (value: string) => void;
	}

	let {
		argument,
		autocompleteIndex = 0,
		isAutocompleteActive = false,
		isLoadingSuggestions = false,
		onBlur,
		onFocus,
		onInput,
		onKeydown,
		onSelectSuggestion,
		suggestions = [],
		value = ''
	}: Props = $props();
</script>

<div class="relative grid gap-1">
	<Label class="mb-1 text-muted-foreground" for="arg-{argument.name}">
		<span>
			{argument.name}

			{#if argument.required}
				<span class="text-destructive">*</span>
			{/if}
		</span>

		{#if isLoadingSuggestions}
			<span class="text-xs text-muted-foreground/50">...</span>
		{/if}
	</Label>

	<Input
		autocomplete="off"
		id="arg-{argument.name}"
		onblur={onBlur}
		onfocus={onFocus}
		oninput={(e) => onInput(e.currentTarget.value)}
		onkeydown={onKeydown}
		placeholder={argument.description || argument.name}
		required={argument.required}
		type="text"
		{value}
	/>

	{#if isAutocompleteActive && suggestions.length > 0}
		<div
			transition:fly={{ duration: 100, y: -5 }}
			class="absolute top-full right-0 left-0 z-10 mt-1 max-h-32 overflow-y-auto rounded-lg border border-border/50 bg-background shadow-lg"
		>
			{#each suggestions as suggestion, i (suggestion)}
				<button
					class="w-full px-3 py-1.5 text-left text-sm hover:bg-accent {i === autocompleteIndex
						? 'bg-accent'
						: ''}"
					onmousedown={() => onSelectSuggestion(suggestion)}
					type="button"
				>
					{suggestion}
				</button>
			{/each}
		</div>
	{/if}
</div>
