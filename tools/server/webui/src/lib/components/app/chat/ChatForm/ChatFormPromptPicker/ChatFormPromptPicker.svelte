<script lang="ts">
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { debounce } from '$lib/utils';
	import { KeyboardKey } from '$lib/enums';
	import type { MCPPromptInfo, GetPromptResult, MCPServerSettingsEntry } from '$lib/types';
	import { SvelteMap } from 'svelte/reactivity';
	import {
		ChatFormPromptPickerList,
		ChatFormPromptPickerHeader,
		ChatFormPromptPickerArgumentForm
	} from '$lib/components/app/chat';
	import * as Popover from '$lib/components/ui/popover';

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
	let selectedIndexBeforeArgumentForm = $state<number | null>(null);

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

			const initialized = await mcpStore.ensureInitialized(perChatOverrides);

			if (!initialized) {
				prompts = [];

				return;
			}

			prompts = await mcpStore.getAllPrompts();
		} catch (error) {
			console.error('[ChatFormPromptPicker] Failed to load prompts:', error);
			prompts = [];
		} finally {
			isLoading = false;
		}
	}

	function handlePromptClick(prompt: MCPPromptInfo) {
		const args = prompt.arguments ?? [];

		if (args.length > 0) {
			selectedIndexBeforeArgumentForm = selectedIndex;
			selectedPrompt = prompt;
			promptArgs = {};
			promptError = null;

			requestAnimationFrame(() => {
				const firstInput = document.querySelector(`#arg-${args[0].name}`) as HTMLInputElement;
				if (firstInput) {
					firstInput.focus();
				}
			});
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
			const result = await mcpStore.getPrompt(prompt.serverName, prompt.name, args);
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
			const result = await mcpStore.getPromptCompletions(
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

		// Handle Escape - return to prompt selection list
		if (event.key === KeyboardKey.ESCAPE) {
			event.preventDefault();
			event.stopPropagation();
			handleCancelArgumentForm();
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
		// Restore the previously selected prompt index
		if (selectedIndexBeforeArgumentForm !== null) {
			selectedIndex = selectedIndexBeforeArgumentForm;
			selectedIndexBeforeArgumentForm = null;
		}
		selectedPrompt = null;
		promptArgs = {};
		promptError = null;
	}

	export function handleKeydown(event: KeyboardEvent): boolean {
		if (!isOpen) return false;

		if (event.key === KeyboardKey.ESCAPE) {
			event.preventDefault();
			if (selectedPrompt) {
				// Return to prompt selection list, keeping the selected prompt active
				handleCancelArgumentForm();
			} else {
				onClose?.();
			}

			return true;
		}

		if (event.key === KeyboardKey.ARROW_DOWN) {
			event.preventDefault();
			if (filteredPrompts.length > 0) {
				selectedIndex = (selectedIndex + 1) % filteredPrompts.length;
			}

			return true;
		}

		if (event.key === KeyboardKey.ARROW_UP) {
			event.preventDefault();
			if (filteredPrompts.length > 0) {
				selectedIndex = selectedIndex === 0 ? filteredPrompts.length - 1 : selectedIndex - 1;
			}

			return true;
		}

		if (event.key === KeyboardKey.ENTER && !selectedPrompt) {
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

<Popover.Root
	bind:open={isOpen}
	onOpenChange={(open) => {
		if (!open) {
			onClose?.();
		}
	}}
>
	<Popover.Trigger class="pointer-events-none absolute inset-0 opacity-0">
		<span class="sr-only">Open prompt picker</span>
	</Popover.Trigger>

	<Popover.Content
		side="top"
		align="start"
		sideOffset={12}
		class="w-[var(--bits-popover-anchor-width)] max-w-none rounded-xl border-border/50 p-0 shadow-xl {className}"
		onkeydown={handleKeydown}
		onOpenAutoFocus={(e) => e.preventDefault()}
	>
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
	</Popover.Content>
</Popover.Root>
