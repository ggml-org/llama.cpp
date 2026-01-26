<script lang="ts">
	import { mcpClient } from '$lib/clients/mcp.client';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { getFaviconUrl, debounce } from '$lib/utils';
	import type { MCPPromptInfo, GetPromptResult, MCPServerSettingsEntry } from '$lib/types';
	import { fly } from 'svelte/transition';
	import { SvelteMap } from 'svelte/reactivity';
	import { SearchInput } from '$lib/components/app';

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
				{@const faviconUrl = server ? getFaviconUrl(server.url) : null}

				<div class="p-4">
					<div class="flex items-start gap-3">
						{#if faviconUrl}
							<img
								src={faviconUrl}
								alt=""
								class="mt-0.5 h-5 w-5 shrink-0 rounded"
								onerror={(e) => {
									(e.currentTarget as HTMLImageElement).style.display = 'none';
								}}
							/>
						{/if}

						<div class="min-w-0 flex-1">
							<div class="text-xs text-muted-foreground">
								{server ? mcpStore.getServerLabel(server) : selectedPrompt.serverName}
							</div>

							<div class="flex items-center gap-2">
								<span class="font-medium">
									{selectedPrompt.title || selectedPrompt.name}
								</span>

								{#if selectedPrompt.arguments?.length}
									<span class="rounded-full bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
										{selectedPrompt.arguments.length} arg{selectedPrompt.arguments.length > 1
											? 's'
											: ''}
									</span>
								{/if}
							</div>
							{#if selectedPrompt.description}
								<p class="mt-1 text-sm text-muted-foreground">
									{selectedPrompt.description}
								</p>
							{/if}
						</div>
					</div>

					<form onsubmit={handleArgumentSubmit} class="space-y-3 pt-4">
						{#each selectedPrompt.arguments ?? [] as arg (arg.name)}
							<div class="relative grid gap-1">
								<label
									for="arg-{arg.name}"
									class="mb-1 flex items-center gap-2 text-sm text-muted-foreground"
								>
									<span>
										{arg.name}
										{#if arg.required}<span class="text-destructive">*</span>{/if}
									</span>
									{#if loadingSuggestions[arg.name]}
										<span class="text-xs text-muted-foreground/50">...</span>
									{/if}
								</label>

								<input
									id="arg-{arg.name}"
									type="text"
									value={promptArgs[arg.name] ?? ''}
									oninput={(e) => handleArgInput(arg.name, e.currentTarget.value)}
									onkeydown={(e) => handleArgKeydown(e, arg.name)}
									onblur={() => handleArgBlur(arg.name)}
									onfocus={() => {
										if ((suggestions[arg.name]?.length ?? 0) > 0) {
											activeAutocomplete = arg.name;
										}
									}}
									placeholder={arg.description || arg.name}
									required={arg.required}
									autocomplete="off"
									class="w-full rounded-lg border border-border/50 bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none"
								/>

								{#if activeAutocomplete === arg.name && (suggestions[arg.name]?.length ?? 0) > 0}
									<div
										class="absolute top-full right-0 left-0 z-10 mt-1 max-h-32 overflow-y-auto rounded-lg border border-border/50 bg-background shadow-lg"
										transition:fly={{ y: -5, duration: 100 }}
									>
										{#each suggestions[arg.name] ?? [] as suggestion, i (suggestion)}
											<button
												type="button"
												onmousedown={() => selectSuggestion(arg.name, suggestion)}
												class="w-full px-3 py-1.5 text-left text-sm hover:bg-accent {i ===
												autocompleteIndex
													? 'bg-accent'
													: ''}"
											>
												{suggestion}
											</button>
										{/each}
									</div>
								{/if}
							</div>
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
								onclick={() => {
									selectedPrompt = null;
									promptArgs = {};
									promptError = null;
								}}
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
				</div>
			{:else}
				<div>
					{#if showSearchInput}
						<div class="p-2 pb-0">
							<SearchInput placeholder="Search prompts..." bind:value={internalSearchQuery} />
						</div>
					{/if}

					<div class="max-h-64 overflow-y-auto p-2">
						{#if isLoading}
							<div class="flex items-center justify-center py-6 text-sm text-muted-foreground">
								Loading prompts...
							</div>
						{:else if filteredPrompts.length === 0}
							<div class="py-6 text-center text-sm text-muted-foreground">
								{prompts.length === 0 ? 'No MCP prompts available' : 'No prompts found'}
							</div>
						{:else}
							{#each filteredPrompts as prompt, index (prompt.serverName + ':' + prompt.name)}
								{@const server = serverSettingsMap.get(prompt.serverName)}
								{@const faviconUrl = server ? getFaviconUrl(server.url) : null}

								<button
									type="button"
									onclick={() => handlePromptClick(prompt)}
									class="flex w-full cursor-pointer items-start gap-3 rounded-lg px-3 py-2 text-left hover:bg-accent {index ===
									selectedIndex
										? 'bg-accent'
										: ''}"
								>
									<div class="min-w-0 flex-1">
										<div class="mb-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
											{#if faviconUrl}
												<img
													src={faviconUrl}
													alt=""
													class="h-3 w-3 shrink-0 rounded-sm"
													onerror={(e) => {
														(e.currentTarget as HTMLImageElement).style.display = 'none';
													}}
												/>
											{/if}

											<span>{server ? mcpStore.getServerLabel(server) : prompt.serverName}</span>
										</div>

										<div class="flex items-center gap-2">
											<span class="font-medium">{prompt.title || prompt.name}</span>

											{#if prompt.arguments && prompt.arguments.length > 0}
												<span class="text-xs text-muted-foreground">
													({prompt.arguments.length} arg{prompt.arguments.length > 1 ? 's' : ''})
												</span>
											{/if}
										</div>

										{#if prompt.description}
											<p class="truncate text-sm text-muted-foreground">
												{prompt.description}
											</p>
										{/if}
									</div>
								</button>
							{/each}
						{/if}
					</div>
				</div>
			{/if}
		</div>
	</div>
{/if}
