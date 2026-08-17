<script lang="ts">
	import { ChevronDown, Clock, FileText, Layers } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
	import { Switch } from '$lib/components/ui/switch';
	import { untrack } from 'svelte';
	import SkillProviderLabel from './SkillProviderLabel.svelte';
	import type { SkillCatalogEntry } from '$lib/types';
	import { normalizeSkillDescription } from '$lib/utils';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		entries: readonly SkillCatalogEntry[];
		selectedId: string | null;
		open: boolean;
		onSelect: (entry: SkillCatalogEntry) => void;
		isDisabled?: (id: string) => boolean;
		onEnabledChange?: (entry: SkillCatalogEntry, enabled: boolean) => void;
	}

	let {
		entries,
		onSelect,
		open,
		selectedId,
		isDisabled = () => false,
		onEnabledChange
	}: Props = $props();

	let expandedDescriptions = new SvelteSet();
	let overflowingDescriptions = new SvelteSet();

	function isDescriptionExpanded(id: string): boolean {
		return expandedDescriptions.has(id);
	}

	function toggleDescription(event: MouseEvent, id: string) {
		event.preventDefault();
		event.stopPropagation();

		if (expandedDescriptions.has(id)) expandedDescriptions.delete(id);
		else expandedDescriptions.add(id);
	}

	function measureDescription(id: string) {
		return (node: HTMLElement) => {
			const observer =
				typeof ResizeObserver === 'undefined'
					? null
					: new ResizeObserver(() => measure(isDescriptionExpanded(id)));

			observer?.observe(node);

			// Re-measure after clamp changes; skip expanded text until collapse.
			$effect(() => {
				const expanded = isDescriptionExpanded(id);
				untrack(() => measure(expanded));
			});

			function measure(expanded: boolean) {
				if (!observer || expanded) return;

				const overflowing = node.scrollHeight > node.clientHeight + 1;
				const alreadyTracked = untrack(() => overflowingDescriptions.has(id));

				if (alreadyTracked === overflowing) return;

				untrack(() => {
					if (overflowing) overflowingDescriptions.add(id);
					else overflowingDescriptions.delete(id);
				});
			}

			return () => {
				observer?.disconnect();
				overflowingDescriptions.delete(id);
			};
		};
	}

	function handleKeydown(event: KeyboardEvent, entry: SkillCatalogEntry) {
		if (event.key !== 'Enter' && event.key !== ' ') return;

		event.preventDefault();
		onSelect(entry);
	}

	function formatTimestamp(value: string | null): string {
		if (!value) return '-';

		const date = new Date(value);

		if (Number.isNaN(date.getTime())) return '-';

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
			class="cursor-pointer transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-hidden hover:bg-accent/50 {open
				? 'me-6'
				: ''}"
			onclick={() => onSelect(entry)}
			onkeydown={(event) => handleKeydown(event, entry)}
		>
			<CardHeader class="flex flex-row items-start justify-between gap-2 space-y-0">
				<div class="flex flex-col gap-1">
					<CardTitle class="text-base">{entry.name}</CardTitle>

					<div class="flex flex-wrap items-center gap-1.5">
						<Badge variant="secondary">{entry.scope}</Badge>
						<Badge variant="outline">
							<SkillProviderLabel provider={entry.provider} />
						</Badge>
						{#if entry.disable_model_invocation}
							<Badge
								variant="secondary"
								title="Not available to the model; activate with /skills <name> or from this catalog."
							>
								Manual only
							</Badge>
						{/if}
						{#if isDisabled(entry.id)}
							<Badge
								variant="secondary"
								title="Not available to the model until re-enabled from this catalog."
							>
								Disabled
							</Badge>
						{/if}
					</div>
				</div>

				{#if onEnabledChange}
					<div
						role="presentation"
						class="flex shrink-0 items-center gap-2"
						onclick={(event) => event.stopPropagation()}
					>
						<label class="text-xs text-muted-foreground" for="skill-enabled-{entry.id}">
							{isDisabled(entry.id) ? 'Enable' : 'Disable'}
						</label>
						<Switch
							id="skill-enabled-{entry.id}"
							checked={!isDisabled(entry.id)}
							aria-label={`${isDisabled(entry.id) ? 'Enable' : 'Disable'} ${entry.name}`}
							onclick={(event) => event.stopPropagation()}
							onkeydown={(event) => event.stopPropagation()}
							onCheckedChange={(enabled) => onEnabledChange(entry, enabled === true)}
						/>
					</div>
				{/if}
			</CardHeader>

			<CardContent class="flex flex-col gap-3">
				{#if entry.description}
					<p
						data-testid="skill-description-{entry.id}"
						class="text-sm text-muted-foreground {!isDescriptionExpanded(entry.id)
							? 'line-clamp-3'
							: ''}"
						{@attach measureDescription(entry.id)}
					>
						{normalizeSkillDescription(entry.description)}
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
						{entry.instruction.tokens_estimated
							? '~'
							: ''}{entry.instruction.tokens.toLocaleString()} tokens
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
