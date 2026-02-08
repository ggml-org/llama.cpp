<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import type { DatabaseMessageExtraMcpPrompt } from '$lib/types';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { SvelteMap } from 'svelte/reactivity';
	import { McpPromptVariant } from '$lib/enums';
	import { TruncatedText } from '$lib/components/app/misc';
	import * as Tooltip from '$lib/components/ui/tooltip';

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
	let textSizeClass = $derived(isAttachment ? 'text-xs' : 'text-md');
	let paddingClass = $derived(isAttachment ? 'px-3 py-2' : 'px-3.75 py-2.5');
	let maxHeightStyle = $derived(
		isAttachment ? 'max-height: 6rem;' : 'max-height: var(--max-message-height);'
	);

	const serverFavicon = $derived(mcpStore.getServerFavicon(prompt.serverName));
	const serverDisplayName = $derived(mcpStore.getServerDisplayName(prompt.serverName));
</script>

<div class="flex flex-col gap-2 {className}">
	<div class="flex items-center justify-between gap-2">
		<div class="inline-flex flex-wrap items-center gap-1.25 text-xs text-muted-foreground">
			<Tooltip.Root>
				<Tooltip.Trigger>
					{#if serverFavicon}
						<img
							src={serverFavicon}
							alt=""
							class="h-3.5 w-3.5 shrink-0 rounded-sm"
							onerror={(e) => {
								(e.currentTarget as HTMLImageElement).style.display = 'none';
							}}
						/>
					{/if}
				</Tooltip.Trigger>

				<Tooltip.Content>
					<span>{serverDisplayName}</span>
				</Tooltip.Content>
			</Tooltip.Root>

			<TruncatedText text={prompt.name} />
		</div>

		{#if showArgBadges}
			<div class="flex flex-wrap justify-end gap-1">
				{#each argumentEntries as [key, value] (key)}
					<Tooltip.Root>
						<Tooltip.Trigger>
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
						</Tooltip.Trigger>
						<Tooltip.Content>
							<span class="max-w-xs break-all">{value}</span>
						</Tooltip.Content>
					</Tooltip.Root>
				{/each}
			</div>
		{/if}
	</div>

	{#if loadError}
		<Card
			class="relative overflow-hidden rounded-[1.125rem] border border-destructive/50 bg-destructive/10 backdrop-blur-md"
		>
			<div
				class="overflow-y-auto {paddingClass}"
				style="{maxHeightStyle} overflow-wrap: anywhere; word-break: break-word;"
			>
				<span class="{textSizeClass} text-destructive">{loadError}</span>
			</div>
		</Card>
	{:else if isLoading}
		<Card
			class="relative overflow-hidden rounded-[1.125rem] border border-purple-200 bg-purple-500/10 px-1 py-2 backdrop-blur-md dark:border-purple-800 dark:bg-purple-500/20"
		>
			<div
				class="overflow-y-auto {paddingClass}"
				style="{maxHeightStyle} overflow-wrap: anywhere; word-break: break-word;"
			>
				<div class="space-y-2">
					<div class="h-3 w-3/4 animate-pulse rounded bg-foreground/20"></div>
					<div class="h-3 w-full animate-pulse rounded bg-foreground/20"></div>
					<div class="h-3 w-5/6 animate-pulse rounded bg-foreground/20"></div>
				</div>
			</div>
		</Card>
	{:else if hasContent}
		<Card
			class="relative overflow-hidden rounded-[1.125rem] border border-purple-200 bg-purple-500/10 py-0 text-foreground backdrop-blur-md dark:border-purple-800 dark:bg-purple-500/20"
		>
			<div
				class="overflow-y-auto {paddingClass}"
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
