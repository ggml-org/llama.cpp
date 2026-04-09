<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { InputWithSuggestions } from '$lib/components/app';
	import { KeyboardKey } from '$lib/enums';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { MIN_AUTOCOMPLETE_INPUT_LENGTH } from '$lib/constants';
	import type { MCPResourceTemplateInfo } from '$lib/types';
	import {
		debounce,
		extractTemplateVariables,
		expandTemplate,
		isTemplateComplete
	} from '$lib/utils';

	interface Props {
		template: MCPResourceTemplateInfo;
		onResolve: (uri: string, serverName: string) => void;
		onCancel: () => void;
	}

	let { template, onResolve, onCancel }: Props = $props();

	const variables = $derived(extractTemplateVariables(template.uriTemplate));

	let values = $state<Record<string, string>>({});
	let suggestions = $state<Record<string, string[]>>({});
	let loadingSuggestions = $state<Record<string, boolean>>({});
	let activeAutocomplete = $state<string | null>(null);
	let autocompleteIndex = $state(0);

	const expandedUri = $derived(expandTemplate(template.uriTemplate, values));
	const isComplete = $derived(isTemplateComplete(template.uriTemplate, values));

	const fetchCompletions = debounce(async (argName: string, value: string) => {
		if (value.length < 1) {
			suggestions[argName] = [];

			return;
		}

		loadingSuggestions[argName] = true;

		try {
			const result = await mcpStore.getResourceCompletions(
				template.serverName,
				template.uriTemplate,
				argName,
				value
			);

			if (result && result.values.length > 0) {
				const filteredValues = result.values.filter((v) => v.trim() !== '');

				if (filteredValues.length > 0) {
					suggestions[argName] = filteredValues;
					activeAutocomplete = argName;
					autocompleteIndex = 0;
				} else {
					suggestions[argName] = [];
				}
			} else {
				suggestions[argName] = [];
			}
		} catch (error) {
			console.error('[McpResourceTemplateForm] Failed to fetch completions:', error);
			suggestions[argName] = [];
		} finally {
			loadingSuggestions[argName] = false;
		}
	}, 200);

	function handleArgInput(argName: string, value: string) {
		values[argName] = value;
		fetchCompletions(argName, value);
	}

	function selectSuggestion(argName: string, value: string) {
		values[argName] = value;
		suggestions[argName] = [];
		activeAutocomplete = null;
	}

	function handleArgKeydown(event: KeyboardEvent, argName: string) {
		const argSuggestions = suggestions[argName] ?? [];

		if (event.key === KeyboardKey.ESCAPE) {
			event.preventDefault();
			event.stopPropagation();

			if (argSuggestions.length > 0 && activeAutocomplete === argName) {
				suggestions[argName] = [];
				activeAutocomplete = null;
			} else {
				onCancel();
			}

			return;
		}

		if (argSuggestions.length === 0 || activeAutocomplete !== argName) return;

		if (event.key === KeyboardKey.ARROW_DOWN) {
			event.preventDefault();
			autocompleteIndex = Math.min(autocompleteIndex + 1, argSuggestions.length - 1);
		} else if (event.key === KeyboardKey.ARROW_UP) {
			event.preventDefault();
			autocompleteIndex = Math.max(autocompleteIndex - 1, 0);
		} else if (event.key === KeyboardKey.ENTER && argSuggestions[autocompleteIndex]) {
			event.preventDefault();
			event.stopPropagation();
			selectSuggestion(argName, argSuggestions[autocompleteIndex]);
		}
	}

	function handleArgBlur(argName: string) {
		setTimeout(() => {
			if (activeAutocomplete === argName) {
				suggestions[argName] = [];
				activeAutocomplete = null;
			}
		}, 150);
	}

	function handleArgFocus(argName: string) {
		const value = values[argName] ?? '';

		if (value.length >= MIN_AUTOCOMPLETE_INPUT_LENGTH) {
			fetchCompletions(argName, value);
		}
	}

	function handleSubmit(event: SubmitEvent) {
		event.preventDefault();

		if (isComplete) {
			onResolve(expandedUri, template.serverName);
		}
	}
</script>

<form onsubmit={handleSubmit} class="space-y-3">
	{#each variables as variable (variable.name)}
		<InputWithSuggestions
			name={variable.name}
			value={values[variable.name] ?? ''}
			suggestions={suggestions[variable.name] ?? []}
			isLoadingSuggestions={loadingSuggestions[variable.name] ?? false}
			isAutocompleteActive={activeAutocomplete === variable.name}
			autocompleteIndex={activeAutocomplete === variable.name ? autocompleteIndex : 0}
			onInput={(value) => handleArgInput(variable.name, value)}
			onKeydown={(e) => handleArgKeydown(e, variable.name)}
			onBlur={() => handleArgBlur(variable.name)}
			onFocus={() => handleArgFocus(variable.name)}
			onSelectSuggestion={(value) => selectSuggestion(variable.name, value)}
		/>
	{/each}

	{#if isComplete}
		<div class="rounded-md bg-muted/50 px-3 py-2">
			<p class="text-xs text-muted-foreground">Resolved URI:</p>

			<p class="mt-0.5 font-mono text-xs break-all">{expandedUri}</p>
		</div>
	{/if}

	<div class="flex justify-end gap-2 pt-1">
		<Button type="button" size="sm" variant="secondary" onclick={onCancel}>Cancel</Button>

		<Button size="sm" type="submit" disabled={!isComplete}>Read Resource</Button>
	</div>
</form>
