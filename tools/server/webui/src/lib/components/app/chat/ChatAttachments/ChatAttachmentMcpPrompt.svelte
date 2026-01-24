<script lang="ts">
	import { ChevronDown, ChevronUp, Eye, Pencil, Check, X } from '@lucide/svelte';
	import type { DatabaseMessageExtraMcpPrompt, MCPServerSettingsEntry } from '$lib/types';
	import { getFaviconUrl, getMcpServerLabel } from '$lib/utils/mcp';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { SvelteMap } from 'svelte/reactivity';
	import { Card } from '$lib/components/ui/card';
	import { MarkdownContent } from '$lib/components/app';
	import { config } from '$lib/stores/settings.svelte';

	interface Props {
		class?: string;
		prompt: DatabaseMessageExtraMcpPrompt;
		readonly?: boolean;
		isLoading?: boolean;
		loadError?: string;
		isEditingAllowed?: boolean;
		onRemove?: () => void;
		onArgumentsChange?: (args: Record<string, string>) => void;
	}

	let {
		class: className = '',
		prompt,
		readonly = false,
		isLoading = false,
		loadError,
		isEditingAllowed = false,
		onRemove,
		onArgumentsChange
	}: Props = $props();

	let showArguments = $state(false);
	let showContent = $state(false);
	let isEditingArguments = $state(false);
	let editedArguments = $state<Record<string, string>>({});
	let hasArguments = $derived(prompt.arguments && Object.keys(prompt.arguments).length > 0);
	let hasContent = $derived(prompt.content && prompt.content.trim().length > 0);

	const currentConfig = config();

	let serverSettingsMap = $derived.by(() => {
		const servers = mcpStore.getServers();
		const map = new SvelteMap<string, MCPServerSettingsEntry>();

		for (const server of servers) {
			map.set(server.id, server);
		}

		return map;
	});

	function getServerFavicon(): string | null {
		const server = serverSettingsMap.get(prompt.serverName);
		return server ? getFaviconUrl(server.url) : null;
	}

	function getServerDisplayName(): string {
		const server = serverSettingsMap.get(prompt.serverName);
		if (!server) return prompt.serverName;

		const healthState = mcpStore.getHealthCheckState(server.id);
		return getMcpServerLabel(server, healthState);
	}

	function toggleArguments(event: MouseEvent) {
		event.stopPropagation();
		showArguments = !showArguments;
	}

	function toggleContent(event: MouseEvent) {
		event.stopPropagation();
		showContent = !showContent;
	}

	function startEditingArguments(event: MouseEvent) {
		event.stopPropagation();
		editedArguments = { ...(prompt.arguments ?? {}) };
		isEditingArguments = true;
		showArguments = true;
	}

	function saveArguments(event: MouseEvent) {
		event.stopPropagation();
		onArgumentsChange?.(editedArguments);
		isEditingArguments = false;
	}

	function cancelEditingArguments(event: MouseEvent) {
		event.stopPropagation();
		isEditingArguments = false;
		editedArguments = {};
	}

	function updateArgument(key: string, value: string) {
		editedArguments = { ...editedArguments, [key]: value };
	}
</script>

<div class="flex flex-col {className}">
	<div
		class="group relative flex flex-col overflow-hidden rounded-lg border bg-gradient-to-br {loadError
			? 'border-destructive/50 from-destructive/10 to-destructive/5 dark:border-destructive/30 dark:from-destructive/20 dark:to-destructive/10'
			: 'border-purple-200 from-purple-50 to-indigo-50 dark:border-purple-800 dark:from-purple-950/50 dark:to-indigo-950/50'}"
	>
		<div class="flex items-start gap-2 p-3">
			<div class="min-w-0 flex-1">
				<div class="mb-1 flex items-center gap-1.5 text-xs text-purple-600 dark:text-purple-400">
					{#if getServerFavicon()}
						<img
							src={getServerFavicon()}
							alt=""
							class="h-3 w-3 shrink-0 rounded-sm"
							onerror={(e) => {
								(e.currentTarget as HTMLImageElement).style.display = 'none';
							}}
						/>
					{/if}

					<span>{getServerDisplayName()}</span>
				</div>

				<div class="font-medium text-foreground">
					{prompt.name}

					{#if isLoading}
						<span class="ml-1 text-xs font-normal text-muted-foreground">Loading...</span>
					{/if}
				</div>

				{#if loadError}
					<p class="mt-1 text-xs text-destructive">{loadError}</p>
				{/if}

				{#if hasContent}
					<button
						type="button"
						onclick={toggleContent}
						class="mt-1 flex items-center gap-1 text-xs text-purple-500 hover:text-purple-700 dark:text-purple-400 dark:hover:text-purple-300"
					>
						{#if showContent}
							<ChevronUp class="h-3 w-3" />
							<span>Hide content</span>
						{:else}
							<Eye class="h-3 w-3" />
							<span>Show content</span>
						{/if}
					</button>
				{/if}

				{#if hasArguments}
					<div class="mt-1 flex items-center gap-2">
						<button
							type="button"
							onclick={toggleArguments}
							class="flex items-center gap-1 text-xs text-purple-500 hover:text-purple-700 dark:text-purple-400 dark:hover:text-purple-300"
						>
							{#if showArguments}
								<ChevronUp class="h-3 w-3" />
								<span>Hide arguments</span>
							{:else}
								<ChevronDown class="h-3 w-3" />
								<span>Show arguments ({Object.keys(prompt.arguments ?? {}).length})</span>
							{/if}
						</button>

						{#if isEditingAllowed && !isEditingArguments && onArgumentsChange}
							<button
								type="button"
								onclick={startEditingArguments}
								class="flex items-center gap-1 text-xs text-purple-500 hover:text-purple-700 dark:text-purple-400 dark:hover:text-purple-300"
								aria-label="Edit arguments"
							>
								<Pencil class="h-3 w-3" />
								<span>Edit</span>
							</button>
						{/if}
					</div>
				{/if}
			</div>

			{#if !readonly && onRemove}
				<button
					type="button"
					onclick={(e) => {
						e.stopPropagation();
						onRemove?.();
					}}
					class="hover:text-destructive-foreground absolute top-1 right-1 flex h-5 w-5 items-center justify-center rounded-full bg-background/80 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 hover:bg-destructive"
					aria-label="Remove prompt"
				>
					<span class="text-xs">×</span>
				</button>
			{/if}
		</div>

		{#if showArguments && hasArguments}
			<div
				class="border-t border-purple-200 bg-purple-50/50 px-3 py-2 dark:border-purple-800 dark:bg-purple-950/30"
			>
				{#if isEditingArguments}
					<div class="space-y-2">
						{#each Object.entries(editedArguments) as [key, value] (key)}
							<div class="flex flex-col gap-1">
								<label
									for="arg-{key}"
									class="text-xs font-medium text-purple-600 dark:text-purple-400"
								>
									{key}
								</label>
								<input
									id="arg-{key}"
									type="text"
									{value}
									oninput={(e) => updateArgument(key, e.currentTarget.value)}
									class="w-full rounded border border-purple-300 bg-white px-2 py-1 text-xs text-foreground outline-none focus:border-purple-500 dark:border-purple-700 dark:bg-purple-950/50"
								/>
							</div>
						{/each}

						<div class="flex justify-end gap-2 pt-1">
							<button
								type="button"
								onclick={cancelEditingArguments}
								class="flex items-center gap-1 rounded px-2 py-1 text-xs text-muted-foreground hover:bg-purple-100 dark:hover:bg-purple-900/50"
							>
								<X class="h-3 w-3" />
								<span>Cancel</span>
							</button>
							<button
								type="button"
								onclick={saveArguments}
								class="flex items-center gap-1 rounded bg-purple-500 px-2 py-1 text-xs text-white hover:bg-purple-600"
							>
								<Check class="h-3 w-3" />
								<span>Save</span>
							</button>
						</div>
					</div>
				{:else}
					<div class="space-y-1">
						{#each Object.entries(prompt.arguments ?? {}) as [key, value] (key)}
							<div class="flex gap-2 text-xs">
								<span class="shrink-0 font-medium text-purple-600 dark:text-purple-400">{key}:</span
								>
								<span class="truncate text-muted-foreground">{value}</span>
							</div>
						{/each}
					</div>
				{/if}
			</div>
		{/if}
	</div>

	{#if showContent && hasContent}
		<Card
			class="mt-2 max-h-64 overflow-y-auto rounded-[1.125rem] border-none bg-purple-500/10 px-3.75 py-2.5 text-foreground backdrop-blur-md dark:bg-purple-500/20"
			style="overflow-wrap: anywhere; word-break: break-word;"
		>
			{#if currentConfig.renderUserContentAsMarkdown}
				<div class="text-sm">
					<MarkdownContent class="markdown-user-content text-foreground" content={prompt.content} />
				</div>
			{:else}
				<span class="text-sm whitespace-pre-wrap">
					{prompt.content}
				</span>
			{/if}
		</Card>
	{/if}
</div>
