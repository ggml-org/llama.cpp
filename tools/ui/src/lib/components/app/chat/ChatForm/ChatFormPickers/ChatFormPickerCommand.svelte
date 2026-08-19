<script lang="ts">
	import { FolderOpen, FoldVertical, Sparkles } from '@lucide/svelte';
	import {
		ChatFormPickerList,
		ChatFormPickerListItem,
		ChatFormPickerPopover
	} from '$lib/components/app/chat';
	import { MODEL_SELECTOR_ICON } from '$lib/constants';
	import { ChatFormCommandAction } from '$lib/enums';
	import { usePickerNavigation } from '$lib/hooks/use-picker-navigation.svelte';
	import type { ChatFormCommand } from '$lib/types';

	/**
	 * Slash-command picker; `query` (typed after `/`) filters the commands.
	 * The parent owns the "dismissed token, don't act until it changes"
	 * snapshot, so this picker just renders and reports selection.
	 */
	interface Props {
		class?: string;
		isOpen: boolean;
		query: string;
		commands: ChatFormCommand[];
		onClose: () => void;
		onSelect: (command: ChatFormCommand) => void;
	}

	let { class: className = '', commands, isOpen, onClose, onSelect, query }: Props = $props();

	const commandIcon: Record<ChatFormCommandAction, typeof Sparkles> = {
		[ChatFormCommandAction.COMPACT]: FoldVertical,
		[ChatFormCommandAction.CWD]: FolderOpen,
		[ChatFormCommandAction.MODEL]: MODEL_SELECTOR_ICON,
		[ChatFormCommandAction.PROMPT]: Sparkles
	};

	const trimmedQuery = $derived((query ?? '').trim().toLowerCase());

	const filteredCommands = $derived(
		trimmedQuery
			? commands.filter(
					(c) =>
						c.name.toLowerCase().includes(trimmedQuery) ||
						c.description.toLowerCase().includes(trimmedQuery) ||
						(c.keywords ?? []).some((k) => k.toLowerCase().includes(trimmedQuery))
				)
			: commands
	);

	function firstEnabledIndex(): number {
		return filteredCommands.findIndex((c) => !c.disabled);
	}

	function stepEnabled(from: number, dir: number): number {
		const n = filteredCommands.length;

		if (n === 0) return -1;

		for (let i = 1; i <= n; i++) {
			const idx = (from + dir * i + n) % n;

			if (!filteredCommands[idx].disabled) return idx;
		}

		return -1;
	}

	const nav = usePickerNavigation({
		count: () => filteredCommands.length,
		isOpen: () => isOpen,
		onClose: () => onClose(),
		onSelect: (index) => handleSelect(filteredCommands[index]),
		step: (from, dir) => (from < 0 ? firstEnabledIndex() : stepEnabled(from, dir))
	});

	$effect(() => {
		if (isOpen) {
			nav.reset(firstEnabledIndex());
		}
	});

	$effect(() => {
		if (nav.hoveredIndex < 0 || nav.hoveredIndex >= filteredCommands.length) {
			nav.reset(firstEnabledIndex());

			return;
		}

		if (filteredCommands[nav.hoveredIndex].disabled) {
			nav.reset(firstEnabledIndex());
		}
	});

	function handleSelect(command: ChatFormCommand) {
		if (command.disabled) return;

		onSelect(command);
		onClose();
	}

	export function handleKeydown(event: KeyboardEvent): boolean {
		return nav.handleKeydown(event);
	}
</script>

<ChatFormPickerPopover
	bind:isOpen
	class={className}
	srLabel="Open command picker"
	{onClose}
	onKeydown={handleKeydown}
>
	<ChatFormPickerList
		items={filteredCommands}
		isLoading={false}
		selectedIndex={nav.hoveredIndex}
		showSearchInput={false}
		searchQuery={query ?? ''}
		emptyMessage="No matching command"
		itemKey={(command) => command.name}
		scrollTrigger={nav.scrollTrigger}
	>
		{#snippet item(command, index, isSelected)}
			{@const Icon = commandIcon[command.action]}
			<ChatFormPickerListItem
				dataIndex={index}
				{isSelected}
				disabled={command.disabled}
				onclick={() => handleSelect(command)}
				onmouseenter={() => {
					if (!command.disabled) nav.setHover(index);
				}}
			>
				<Icon class="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
				<div class="flex min-w-0 flex-1 flex-col">
					<span class="font-mono text-sm font-medium">/{command.name}</span>
					<span class="min-w-0 flex-1 truncate text-left text-xs text-muted-foreground">
						{command.description}
					</span>
				</div>
			</ChatFormPickerListItem>
		{/snippet}
	</ChatFormPickerList>
</ChatFormPickerPopover>
