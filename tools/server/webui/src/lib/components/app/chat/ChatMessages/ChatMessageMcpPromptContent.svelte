<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import type { DatabaseMessageExtraMcpPrompt, MCPServerSettingsEntry } from '$lib/types';
	import { getFaviconUrl } from '$lib/utils';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { SvelteMap } from 'svelte/reactivity';
	import { McpPromptVariant } from '$lib/enums';

	interface ContentPart {
		text: string;
		argKey: string | null;
	}

	interface Props {
		class?: string;
		prompt: DatabaseMessageExtraMcpPrompt;
		variant?: McpPromptVariant;
		isLoading?: boolean;
		loadError?: string;
	}

	let {
		class: className = '',
		prompt,
		variant = McpPromptVariant.MESSAGE,
		isLoading = false,
		loadError
	}: Props = $props();

	let hoveredArgKey = $state<string | null>(null);
	let argumentEntries = $derived(Object.entries(prompt.arguments ?? {}));
	let hasArguments = $derived(prompt.arguments && Object.keys(prompt.arguments).length > 0);
	let hasContent = $derived(prompt.content && prompt.content.trim().length > 0);

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

		return mcpStore.getServerLabel(server);
	}

	let contentParts = $derived.by((): ContentPart[] => {
		if (!prompt.content || !hasArguments) {
			return [{ text: prompt.content || '', argKey: null }];
		}

		const parts: ContentPart[] = [];
		let remaining = prompt.content;

		const valueToKey = new SvelteMap<string, string>();
		for (const [key, value] of argumentEntries) {
			if (value && value.trim()) {
				valueToKey.set(value, key);
			}
		}

		const sortedValues = [...valueToKey.keys()].sort((a, b) => b.length - a.length);

		while (remaining.length > 0) {
			let earliestMatch: { index: number; value: string; key: string } | null = null;

			for (const value of sortedValues) {
				const index = remaining.indexOf(value);
				if (index !== -1 && (earliestMatch === null || index < earliestMatch.index)) {
					earliestMatch = { index, value, key: valueToKey.get(value)! };
				}
			}

			if (earliestMatch) {
				if (earliestMatch.index > 0) {
					parts.push({ text: remaining.slice(0, earliestMatch.index), argKey: null });
				}

				parts.push({ text: earliestMatch.value, argKey: earliestMatch.key });
				remaining = remaining.slice(earliestMatch.index + earliestMatch.value.length);
			} else {
				parts.push({ text: remaining, argKey: null });

				break;
			}
		}

		return parts;
	});

	let showArgBadges = $derived(hasArguments && !isLoading && !loadError);
	let isAttachment = $derived(variant === McpPromptVariant.ATTACHMENT);
	let textSizeClass = $derived(isAttachment ? 'text-sm' : 'text-md');
	let maxHeightStyle = $derived(
		isAttachment ? 'max-height: 10rem;' : 'max-height: var(--max-message-height);'
	);
</script>

<div class="flex flex-col gap-2 {className}">
	<div class="flex items-center justify-between gap-2">
		<div class="flex items-center gap-1.5 text-xs text-muted-foreground">
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
			<span>·</span>
			<span>{prompt.name}</span>
		</div>

		{#if showArgBadges}
			<div class="flex flex-wrap justify-end gap-1">
				{#each argumentEntries as [key] (key)}
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					<span
						class="rounded-sm bg-purple-200/60 px-1.5 py-0.5 text-[10px] leading-none text-purple-700 transition-opacity dark:bg-purple-800/40 dark:text-purple-300 {hoveredArgKey &&
						hoveredArgKey !== key
							? 'opacity-30'
							: ''}"
						onmouseenter={() => (hoveredArgKey = key)}
						onmouseleave={() => (hoveredArgKey = null)}
					>
						{key}
					</span>
				{/each}
			</div>
		{/if}
	</div>

	{#if loadError}
		<Card
			class="relative overflow-hidden rounded-[1.125rem] border border-destructive/50 bg-destructive/10 backdrop-blur-md"
		>
			<div
				class="overflow-y-auto px-3.75 py-2.5"
				style="{maxHeightStyle} overflow-wrap: anywhere; word-break: break-word;"
			>
				<span class="{textSizeClass} text-destructive">{loadError}</span>
			</div>
		</Card>
	{:else if isLoading}
		<Card
			class="relative overflow-hidden rounded-[1.125rem] border border-purple-200 bg-purple-500/10 text-foreground backdrop-blur-md dark:border-purple-800 dark:bg-purple-500/20"
		>
			<div
				class="overflow-y-auto px-3.75 py-2.5"
				style="{maxHeightStyle} overflow-wrap: anywhere; word-break: break-word;"
			>
				<span class="{textSizeClass} text-purple-500 italic dark:text-purple-400">
					Loading prompt content...
				</span>
			</div>
		</Card>
	{:else if hasContent}
		<Card
			class="relative overflow-hidden rounded-[1.125rem] border border-purple-200 bg-purple-500/10 py-0 text-foreground backdrop-blur-md dark:border-purple-800 dark:bg-purple-500/20"
		>
			<div
				class="overflow-y-auto px-3.75 py-2.5"
				style="{maxHeightStyle} overflow-wrap: anywhere; word-break: break-word;"
			>
				<span class="{textSizeClass} whitespace-pre-wrap">
					<!-- This formatting is needed to keep the text in proper shape -->
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					{#each contentParts as part, i (i)}{#if part.argKey}<span
								class="rounded-sm bg-purple-300/50 px-0.5 text-purple-900 transition-opacity dark:bg-purple-700/50 dark:text-purple-100 {hoveredArgKey &&
								hoveredArgKey !== part.argKey
									? 'opacity-30'
									: ''}"
								onmouseenter={() => (hoveredArgKey = part.argKey)}
								onmouseleave={() => (hoveredArgKey = null)}>{part.text}</span
							>{:else}<span class="transition-opacity {hoveredArgKey ? 'opacity-30' : ''}"
								>{part.text}</span
							>{/if}{/each}</span
				>
			</div>
		</Card>
	{/if}
</div>
