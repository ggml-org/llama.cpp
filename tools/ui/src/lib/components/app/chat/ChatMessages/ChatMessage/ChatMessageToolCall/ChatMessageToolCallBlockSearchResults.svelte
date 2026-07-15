<script lang="ts">
	import { ICON_CLASS_DEFAULT, ICON_CLASS_SPIN } from '$lib/constants/css-classes';
	import { Globe, Loader2 } from '@lucide/svelte';
	import { CollapsibleContentBlock } from '$lib/components/app';
	import * as HoverCard from '$lib/components/ui/hover-card';
	import { AgenticSectionType } from '$lib/enums';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import {
		extractSearchResults,
		extractSearchQuery,
		faviconForUrl,
		sanitizeExternalUrl,
		type SearchResult,
		type AgenticSection
	} from '$lib/utils';

	interface Props {
		section: AgenticSection;
		open?: boolean;
		isStreaming?: boolean;
		onToggle?: () => void;
	}

	let { section, open = $bindable(false), isStreaming = false, onToggle }: Props = $props();

	const isPending = $derived(section.type === AgenticSectionType.TOOL_CALL_PENDING);
	const isStreamingCall = $derived(section.type === AgenticSectionType.TOOL_CALL_STREAMING);
	const showSpinner = $derived(isPending || (isStreamingCall && isStreaming));

	const results: SearchResult[] = $derived(extractSearchResults(section.toolResult));
	const query = $derived(extractSearchQuery(section.toolArgs));

	// Same icon-resolution chain as ChatMessageToolCallBlockDefault so
	// MCP-server branding is consistent across both views. Spinner wins
	// while the call is in flight so the user sees execution status.
	const iconUrl = $derived(showSpinner ? null : mcpStore.getServerFaviconForTool(section.toolName));
	const icon = $derived(showSpinner ? Loader2 : undefined);
	const iconClass = $derived(showSpinner ? ICON_CLASS_SPIN : ICON_CLASS_DEFAULT);

	// Verb reflects state: "Searching" while the call is in flight, "Searched"
	// once results (or a definitive empty response) have arrived. Lets the
	// heading read as a live progress indicator rather than a completed
	// retrospective.
	const title = $derived.by(() => {
		const verb = showSpinner ? 'Searching' : 'Searched';
		return query ? `${verb} web for "${query}"` : `${verb} web`;
	});

	function hideBrokenIcon(event: Event) {
		(event.currentTarget as HTMLImageElement).style.display = 'none';
	}

	function formatPublishDate(iso: string | undefined): string | null {
		if (!iso) return null;
		try {
			const date = new Date(iso);
			if (Number.isNaN(date.getTime())) return iso;
			return date.toLocaleDateString(undefined, {
				year: 'numeric',
				month: 'short',
				day: 'numeric'
			});
		} catch {
			return iso;
		}
	}

	function hostFor(url: string): string | null {
		try {
			return new URL(url).host;
		} catch {
			return null;
		}
	}

	function hasDetails(result: SearchResult): boolean {
		return Boolean(result.highlights || result.published || result.author);
	}
</script>

{#snippet pill(result: SearchResult)}
	{@const faviconUrl = faviconForUrl(result.url)}
	{@const safeUrl = sanitizeExternalUrl(result.url)}
	{@const showHoverCard = safeUrl !== null && hasDetails(result)}
	{#if safeUrl}
		<HoverCard.Root openDelay={150} closeDelay={100}>
			<HoverCard.Trigger
				href={safeUrl}
				target="_blank"
				rel="noopener noreferrer"
				class="hover:bg-muted/80 focus-visible:ring-ring inline-flex max-w-full items-center gap-1.5 rounded-full border bg-muted px-2.5 py-1 text-xs transition-colors outline-none focus-visible:ring-2"
			>
				{#if faviconUrl}
					<img
						src={faviconUrl}
						alt=""
						class="h-3 w-3 shrink-0 rounded-sm"
						onerror={hideBrokenIcon}
					/>
				{:else}
					<Globe class="text-muted-foreground/70 h-3 w-3 shrink-0" />
				{/if}
				<span class="truncate font-medium text-foreground/80">{result.title}</span>
			</HoverCard.Trigger>
			{#if showHoverCard}
				{@const publishDate = formatPublishDate(result.published)}
				{@const host = hostFor(safeUrl)}
				<HoverCard.Content
					side="top"
					align="start"
					sideOffset={6}
					class="bg-popover text-popover-foreground z-50 w-80 max-w-[90vw] rounded-lg border p-0 shadow-lg"
				>
					<div class="flex flex-col gap-2 p-3">
						<a
							href={safeUrl}
							target="_blank"
							rel="noopener noreferrer"
							class="line-clamp-3 text-sm font-medium leading-snug hover:underline"
							>{result.title}</a
						>
						{#if publishDate || result.author}
							<div class="text-muted-foreground flex items-center gap-1.5 text-[11px]">
								{#if publishDate}
									<span>{publishDate}</span>
								{/if}
								{#if publishDate && result.author}
									<span class="opacity-50">&middot;</span>
								{/if}
								{#if result.author}
									<span class="truncate">{result.author}</span>
								{/if}
							</div>
						{/if}
						{#if result.highlights}
							<p
								class="text-popover-foreground/85 line-clamp-5 text-xs leading-relaxed whitespace-pre-line"
							>
								{result.highlights}
							</p>
						{/if}
						{#if host}
							<div class="text-muted-foreground/80 truncate text-[11px]">{host}</div>
						{/if}
					</div>
				</HoverCard.Content>
			{/if}
		</HoverCard.Root>
	{/if}
{/snippet}

<CollapsibleContentBlock {open} class="my-2" {icon} {iconClass} {iconUrl} {title} {onToggle}>
	{#if results.length > 0}
		<div class="flex flex-wrap items-center gap-2 pb-1">
			{#each results as result (result.url)}
				{@render pill(result)}
			{/each}
		</div>
	{:else if showSpinner}
		<div class="text-muted-foreground/70 flex items-center gap-2 py-1 text-xs italic">
			<Loader2 class="h-3 w-3 animate-spin" />
			<span>Searching...</span>
		</div>
	{:else}
		<div class="text-muted-foreground/70 py-1 text-xs italic">No results</div>
	{/if}
</CollapsibleContentBlock>
