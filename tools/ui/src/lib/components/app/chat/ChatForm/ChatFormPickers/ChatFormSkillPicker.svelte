<script lang="ts">
	import {
		ChatFormPickerList,
		ChatFormPickerListItem,
		ChatFormPickerPopover
	} from '$lib/components/app/chat';
	import { usePickerNavigation } from '$lib/hooks/use-picker-navigation.svelte';
	import type { SkillCatalogEntry } from '$lib/types';
	import { normalizeSkillDescription } from '$lib/utils';

	/**
	 * Catalog-name picker for `/skills <name>`: the composer is the sole
	 * query control, so there is no search input. Rows expose only the safe
	 * display facts (`name`, `description`, `scope`, `provider`) and both
	 * pointer and keyboard selection report the exact `skill.name`. This
	 * component is data-only: it never fetches, calls `SkillsService`,
	 * resolves names, or renders raw catalog XML.
	 */
	interface Props {
		class?: string;
		isOpen: boolean;
		query: string;
		skills: SkillCatalogEntry[];
		onClose: () => void;
		onSelect: (name: string) => void;
	}

	let { class: className = '', isOpen, onClose, onSelect, query, skills }: Props = $props();

	const normalizedQuery = $derived(query.trim().toLowerCase());
	const matchingSkills = $derived.by(() => {
		const prefix: SkillCatalogEntry[] = [];
		const substring: SkillCatalogEntry[] = [];

		for (const skill of skills) {
			const name = skill.name.toLowerCase();

			if (name.startsWith(normalizedQuery)) prefix.push(skill);
			else if (name.includes(normalizedQuery)) substring.push(skill);
		}

		return [...prefix, ...substring];
	});

	function firstIndex(): number {
		return matchingSkills.length > 0 ? 0 : -1;
	}

	const nav = usePickerNavigation({
		count: () => matchingSkills.length,
		isOpen: () => isOpen,
		onClose: () => onClose(),
		onSelect: (index) => handleSelect(matchingSkills[index])
	});

	$effect(() => {
		if (isOpen) {
			nav.reset(firstIndex());
		}
	});

	$effect(() => {
		if (nav.hoveredIndex < 0 || nav.hoveredIndex >= matchingSkills.length) {
			nav.reset(firstIndex());

			return;
		}
	});

	function handleSelect(skill: SkillCatalogEntry) {
		onSelect(skill.name);
		onClose();
	}

	export function handleKeydown(event: KeyboardEvent): boolean {
		return nav.handleKeydown(event);
	}
</script>

<ChatFormPickerPopover
	bind:isOpen
	class={className}
	srLabel="Open skill picker"
	{onClose}
	onKeydown={handleKeydown}
>
	<ChatFormPickerList
		items={matchingSkills}
		isLoading={false}
		selectedIndex={nav.hoveredIndex}
		showSearchInput={false}
		searchQuery={query ?? ''}
		emptyMessage="No matching skills"
		itemKey={(skill) => skill.id}
		scrollTrigger={nav.scrollTrigger}
	>
		{#snippet item(skill, index, isSelected)}
			<ChatFormPickerListItem
				dataIndex={index}
				{isSelected}
				onclick={() => handleSelect(skill)}
				onmouseenter={() => nav.setHover(index)}
			>
				<div class="flex min-w-0 flex-1 flex-col">
					<span class="font-mono text-sm font-medium">{skill.name}</span>
					<span class="min-w-0 flex-1 truncate text-left text-xs text-muted-foreground">
						{normalizeSkillDescription(skill.description)}
					</span>
					<span class="text-[10px] uppercase tracking-wide text-muted-foreground/70">
						{skill.scope} · {skill.provider}
					</span>
				</div>
			</ChatFormPickerListItem>
		{/snippet}
	</ChatFormPickerList>
</ChatFormPickerPopover>
