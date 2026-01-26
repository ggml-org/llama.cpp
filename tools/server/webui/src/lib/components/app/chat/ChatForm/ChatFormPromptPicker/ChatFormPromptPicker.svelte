<script lang="ts">
	import { mcpClient } from '$lib/clients/mcp.client';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { debounce } from '$lib/utils';
	import type { MCPPromptInfo, GetPromptResult, MCPServerSettingsEntry } from '$lib/types';
	import { fly } from 'svelte/transition';
	import { SvelteMap } from 'svelte/reactivity';
	import ChatFormPromptPickerList from './ChatFormPromptPickerList.svelte';
	import ChatFormPromptPickerHeader from './ChatFormPromptPickerHeader.svelte';
	import ChatFormPromptPickerArgumentForm from './ChatFormPromptPickerArgumentForm.svelte';

	interface Props {
		class?: string;
		isOpen?: boolean;
		searchQuery?: string;
		onClose?: () => void;
		onPromptLoadStart?: (
			placeholderId: string,
			promptInfo: MCPPromptInfo,
			args?: Record<string, string>
		) => void;
		onPromptLoadComplete?: (placeholderId: string, result: GetPromptResult) => void;
		onPromptLoadError?: (placeholderId: string, error: string) => void;
	}

	let {
		class: className = '',
		isOpen = false,
		searchQuery = '',
		onClose,
		onPromptLoadStart,
		onPromptLoadComplete,
		onPromptLoadError
	}: Props = $props();

	let prompts = $state<MCPPromptInfo[]>([]);
	let isLoading = $state(false);
	let selectedPrompt = $state<MCPPromptInfo | null>(null);
	let promptArgs = $state<Record<string, string>>({});
	let selectedIndex = $state(0);
	let internalSearchQuery = $state('');
	let promptError = $state<string | null>(null);

	let suggestions = $state<Record<string, string[]>>({});
	let loadingSuggestions = $state<Record<string, boolean>>({});
	let activeAutocomplete = $state<string | null>(null);
	let autocompleteIndex = $state(0);

	let serverSettingsMap = $derived.by(() => {
		const servers = mcpStore.getServers();
		const map = new SvelteMap<string, MCPServerSettingsEntry>();

		for (const server of servers) {
			map.set(server.id, server);
		}

		return map;
	});

	$effect(() => {
		if (isOpen) {
			loadPrompts();
			selectedIndex = 0;
		} else {
			selectedPrompt = null;
			promptArgs = {};
			promptError = null;
		}
	});

	$effect(() => {
		if (filteredPrompts.length > 0 && selectedIndex >= filteredPrompts.length) {
			selectedIndex = 0;
		}
	});

	async function loadPrompts() {
		isLoading = true;

		try {
			const perChatOverrides = conversationsStore.getAllMcpServerOverrides();

			const initialized = await mcpClient.ensureInitialized(perChatOverrides);

			if (!initialized) {
				prompts = [];
				return;
			}

			prompts = await mcpClient.getAllPrompts();
		} catch (error) {
			console.error('[ChatFormPromptPicker] Failed to load prompts:', error);
			prompts = [];
		} finally {
			isLoading = false;
		}
	}

	function handlePromptClick(prompt: MCPPromptInfo) {
		const requiredArgs = prompt.arguments?.filter((arg) => arg.required) ?? [];

		if (requiredArgs.length > 0) {
			selectedPrompt = prompt;
			promptArgs = {};
			promptError = null;
		} else {
			executePrompt(prompt, {});
		}
	}

	async function executePrompt(prompt: MCPPromptInfo, args: Record<string, string>) {
		promptError = null;

		const placeholderId = crypto.randomUUID();

		const nonEmptyArgs = Object.fromEntries(
			Object.entries(args).filter(([, value]) => value.trim() !== '')
		);
		const argsToPass = Object.keys(nonEmptyArgs).length > 0 ? nonEmptyArgs : undefined;

		onPromptLoadStart?.(placeholderId, prompt, argsToPass);
		onClose?.();

		try {
			const result = await mcpClient.getPrompt(prompt.serverName, prompt.name, args);
			onPromptLoadComplete?.(placeholderId, result);
		} catch (error) {
			const errorMessage =
				error instanceof Error ? error.message : 'Unknown error executing prompt';
			onPromptLoadError?.(placeholderId, errorMessage);
		}
	}

	function handleArgumentSubmit(event: SubmitEvent) {
		event.preventDefault();

		if (selectedPrompt) {
			executePrompt(selectedPrompt, promptArgs);
		}
	}

	const fetchCompletions = debounce(async (argName: string, value: string) => {
		if (!selectedPrompt || value.length < 1) {
			suggestions[argName] = [];
			return;
		}

		if (import.meta.env.DEV) {
			console.log('[ChatFormPromptPicker] Fetching completions for:', {
				serverName: selectedPrompt.serverName,
				promptName: selectedPrompt.name,
				argName,
				value
			});
		}

		loadingSuggestions[argName] = true;

		try {
			const result = await mcpClient.getPromptCompletions(
				selectedPrompt.serverName,
				selectedPrompt.name,
				argName,
				value
			);

			if (import.meta.env.DEV) {
				console.log('[ChatFormPromptPicker] Autocomplete result:', {
					argName,
					value,
					result,
					suggestionsCount: result?.values.length ?? 0
				});
			}

			if (result && result.values.length > 0) {
				// Filter out empty strings from suggestions
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
			console.error('[ChatFormPromptPicker] Failed to fetch completions:', error);
			suggestions[argName] = [];
		} finally {
			loadingSuggestions[argName] = false;
		}
	}, 200);

	function handleArgInput(argName: string, value: string) {
		promptArgs[argName] = value;
		fetchCompletions(argName, value);
	}

	function selectSuggestion(argName: string, value: string) {
		promptArgs[argName] = value;
		suggestions[argName] = [];
		activeAutocomplete = null;
	}

	function handleArgKeydown(event: KeyboardEvent, argName: string) {
		const argSuggestions = suggestions[argName] ?? [];

		if (argSuggestions.length === 0 || activeAutocomplete !== argName) return;

		if (event.key === 'ArrowDown') {
			event.preventDefault();
			autocompleteIndex = Math.min(autocompleteIndex + 1, argSuggestions.length - 1);
		} else if (event.key === 'ArrowUp') {
			event.preventDefault();
			autocompleteIndex = Math.max(autocompleteIndex - 1, 0);
		} else if (event.key === 'Enter' && argSuggestions[autocompleteIndex]) {
			event.preventDefault();
			event.stopPropagation();
			selectSuggestion(argName, argSuggestions[autocompleteIndex]);
		} else if (event.key === 'Escape') {
			event.preventDefault();
			suggestions[argName] = [];
			activeAutocomplete = null;
		}
	}

	function handleArgBlur(argName: string) {
		// Delay to allow click on suggestion
		setTimeout(() => {
			if (activeAutocomplete === argName) {
				suggestions[argName] = [];
				activeAutocomplete = null;
			}
		}, 150);
	}

	function handleArgFocus(argName: string) {
		if ((suggestions[argName]?.length ?? 0) > 0) {
			activeAutocomplete = argName;
		}
	}

	function handleCancelArgumentForm() {
		selectedPrompt = null;
		promptArgs = {};
		promptError = null;
	}

	export function handleKeydown(event: KeyboardEvent): boolean {
		if (!isOpen) return false;

		if (event.key === 'Escape') {
			event.preventDefault();
			if (selectedPrompt) {
				selectedPrompt = null;
				promptArgs = {};
			} else {
				onClose?.();
			}

			return true;
		}

		if (event.key === 'ArrowDown') {
			event.preventDefault();
			if (selectedIndex < filteredPrompts.length - 1) {
				selectedIndex++;
			}

			return true;
		}

		if (event.key === 'ArrowUp') {
			event.preventDefault();
			if (selectedIndex > 0) {
				selectedIndex--;
			}

			return true;
		}

		if (event.key === 'Enter' && !selectedPrompt) {
			event.preventDefault();
			if (filteredPrompts[selectedIndex]) {
				handlePromptClick(filteredPrompts[selectedIndex]);
			}

			return true;
		}

		return false;
	}

	let filteredPrompts = $derived.by(() => {
		const sortedServers = mcpStore.getServersSorted();
		const serverOrderMap = new Map(sortedServers.map((server, index) => [server.id, index]));

		const sortedPrompts = [...prompts].sort((a, b) => {
			const orderA = serverOrderMap.get(a.serverName) ?? Number.MAX_SAFE_INTEGER;
			const orderB = serverOrderMap.get(b.serverName) ?? Number.MAX_SAFE_INTEGER;
			return orderA - orderB;
		});

		const query = (searchQuery || internalSearchQuery).toLowerCase();
		if (!query) return sortedPrompts;

		return sortedPrompts.filter(
			(prompt) =>
				prompt.name.toLowerCase().includes(query) ||
				prompt.title?.toLowerCase().includes(query) ||
				prompt.description?.toLowerCase().includes(query)
		);
	});

	let showSearchInput = $derived(prompts.length > 3);
</script>

{#if isOpen}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class="absolute right-0 bottom-full left-0 z-50 mb-3 {className}"
		transition:fly={{ y: 10, duration: 150 }}
		onkeydown={handleKeydown}
	>
		<div class="overflow-hidden rounded-xl border border-border/50 bg-popover shadow-xl">
			{#if selectedPrompt}
				{@const server = serverSettingsMap.get(selectedPrompt.serverName)}
				{@const serverLabel = server ? mcpStore.getServerLabel(server) : selectedPrompt.serverName}

				<div class="p-4">
					<ChatFormPromptPickerHeader prompt={selectedPrompt} {server} {serverLabel} />

					<ChatFormPromptPickerArgumentForm
						prompt={selectedPrompt}
						{promptArgs}
						{suggestions}
						{loadingSuggestions}
						{activeAutocomplete}
						{autocompleteIndex}
						{promptError}
						onArgInput={handleArgInput}
						onArgKeydown={handleArgKeydown}
						onArgBlur={handleArgBlur}
						onArgFocus={handleArgFocus}
						onSelectSuggestion={selectSuggestion}
						onSubmit={handleArgumentSubmit}
						onCancel={handleCancelArgumentForm}
					/>
				</div>
			{:else}
				<ChatFormPromptPickerList
					prompts={filteredPrompts}
					{isLoading}
					{selectedIndex}
					bind:searchQuery={internalSearchQuery}
					{showSearchInput}
					{serverSettingsMap}
					getServerLabel={(server) => mcpStore.getServerLabel(server)}
					onPromptClick={handlePromptClick}
				/>
			{/if}
		</div>
	</div>
{/if}
