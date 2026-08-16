<script lang="ts">
	import { ChevronDown, Clock, FileText, Layers } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
	import type { SkillCatalogEntry } from '$lib/types';

	interface Props {
		entries: readonly SkillCatalogEntry[];
		selectedId: string | null;
		open: boolean;
		onSelect: (entry: SkillCatalogEntry) => void;
	}

	let { entries, selectedId, open, onSelect }: Props = $props();

	let expandedDescriptions = $state<Set<string>>(new Set());
	let overflowingDescriptions = $state<Set<string>>(new Set());

	function isDescriptionExpanded(id: string): boolean {
		return expandedDescriptions.has(id);
	}

	function toggleDescription(event: MouseEvent, id: string) {
		event.preventDefault();
		event.stopPropagation();

		const next = new Set(expandedDescriptions);
		if (next.has(id)) next.delete(id);
		else next.add(id);
		expandedDescriptions = next;
	}

	function measureDescription(
		node: HTMLElement,
		params: { id: string; expanded: boolean }
	) {
		let current = params;
		const observer =
			typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(measure);

		function measure() {
			if (!observer || current.expanded) return;

			const overflowing = node.scrollHeight > node.clientHeight + 1;
			const alreadyTracked = overflowingDescriptions.has(current.id);
			if (alreadyTracked === overflowing) return;

			const next = new Set(overflowingDescriptions);
			if (overflowing) next.add(current.id);
			else next.delete(current.id);
			overflowingDescriptions = next;
		}

		observer?.observe(node);
		measure();

		return {
			update(next: { id: string; expanded: boolean }) {
				current = next;
				measure();
			},
			destroy() {
				observer?.disconnect();
				const next = new Set(overflowingDescriptions);
				next.delete(current.id);
				overflowingDescriptions = next;
			}
		};
	}

	function handleKeydown(event: KeyboardEvent, entry: SkillCatalogEntry) {
		if (event.key !== 'Enter' && event.key !== ' ') return;

		event.preventDefault();
		onSelect(entry);
	}

	function formatTimestamp(value: string | null): string {
		if (!value) return '—';

		const date = new Date(value);

		if (Number.isNaN(date.getTime())) return '—';

		return date.toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' });
	}

	function resourceLabel(entry: SkillCatalogEntry): string {
		return entry.resources.truncated ? `${entry.resources.count}+` : `${entry.resources.count}`;
	}
</script>

<div class="flex h-full min-h-0 flex-col gap-4 overflow-y-auto">
	{#each entries as entry (entry.id)}
		<Card
			role="button"
			tabindex={0}
			aria-pressed={entry.id === selectedId}
			class="cursor-pointer transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-hidden hover:bg-accent/50 {open ? 'me-6' : ''}"
			onclick={() => onSelect(entry)}
			onkeydown={(event) => handleKeydown(event, entry)}
		>
			<CardHeader class="flex-row items-start justify-between gap-2 space-y-0">
				<div class="flex flex-col gap-1">
					<CardTitle class="text-base">{entry.name}</CardTitle>

					<div class="flex flex-wrap items-center gap-1.5">
						<Badge variant="secondary">{entry.scope}</Badge>
						<Badge variant="outline">{entry.provider}</Badge>
						<Badge variant={entry.instruction.tokens_estimated ? 'tertiary' : 'outline'}>
							{entry.instruction.tokens_estimated ? 'estimated' : 'exact'}
						</Badge>
					</div>
				</div>
			</CardHeader>

			<CardContent class="flex flex-col gap-3">
				{#if entry.description}
					<p
						data-testid="skill-description-{entry.id}"
						class="text-sm text-muted-foreground {!isDescriptionExpanded(entry.id)
							? 'line-clamp-3'
							: ''}"
						use:measureDescription={{
							id: entry.id,
							expanded: isDescriptionExpanded(entry.id)
						}}
					>
						{entry.description}
					</p>

					{#if overflowingDescriptions.has(entry.id)}
						<button
							type="button"
							data-testid="skill-description-toggle-{entry.id}"
							class="inline-flex w-fit items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
							aria-expanded={isDescriptionExpanded(entry.id)}
							onclick={(event) => toggleDescription(event, entry.id)}
							onkeydown={(event) => event.stopPropagation()}
						>
							{isDescriptionExpanded(entry.id) ? 'Show less' : 'Show more'}
							<ChevronDown
								class="size-3.5 transition-transform duration-200 {isDescriptionExpanded(entry.id)
									? 'rotate-180'
									: ''}"
								aria-hidden="true"
							/>
						</button>
					{/if}
				{/if}

				<div class="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
					<span class="inline-flex items-center gap-1">
						<FileText class="h-3 w-3" />
						{entry.instruction.tokens.toLocaleString()} tokens
					</span>
					<span>{entry.instruction.lines.toLocaleString()} lines</span>
					<span>{entry.instruction.bytes.toLocaleString()} bytes</span>
					<span class="inline-flex items-center gap-1">
						<Clock class="h-3 w-3" />
						Modified: {formatTimestamp(entry.instruction.modified_at)}
					</span>
					<span class="inline-flex items-center gap-1">
						<Layers class="h-3 w-3" />
						Resources:
						{resourceLabel(entry)}
					</span>
				</div>
			</CardContent>
		</Card>
	{/each}
</div>
