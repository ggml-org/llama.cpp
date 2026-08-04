<script lang="ts">
	import { FolderOpen, Sparkles } from '@lucide/svelte';
	import { KeyboardKey } from '$lib/enums';
	import { MODEL_SELECTOR_ICON } from '$lib/constants';
	import type { ChatFormCommand, ChatFormCommandAction } from '$lib/types';
	import {
		ChatFormPickerList,
		ChatFormPickerListItem,
		ChatFormPickerPopover
	} from '$lib/components/app/chat';

	/**
	 * Slash-command picker.
	 *
	 * Opens when the user types `/` at the start of the chat input. The
	 * chat input is the search surface: `query` (what the user typed after
	 * `/`) filters the available commands by name/description. Selecting a
	 * command hands it to the parent via `onSelect` so the parent can
	 * dispatch the corresponding picker / selector.
	 *
	 * The parent owns the "user dismissed this token, don't act until it
	 * changes" snapshot, so this picker stays simple - it just renders and
	 * reports selection.
	 */
	interface Props {
		class?: string;
		isOpen: boolean;
		query: string;
		commands: ChatFormCommand[];
		onClose: () => void;
		onSelect: (command: ChatFormCommand) => void;
	}

	let { class: className = '', isOpen, query, commands, onClose, onSelect }: Props = $props();

	let hoveredIndex = $state(-1);
	// Bump on ArrowUp/ArrowDown only; the list's auto-scroll never fires on
	// hover (see `scrollTrigger` prop on ChatFormPickerList).
	let scrollTrigger = $state(0);

	const commandIcon: Record<ChatFormCommandAction, typeof Sparkles> = {
		prompt: Sparkles,
		cwd: FolderOpen,
		model: MODEL_SELECTOR_ICON
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

	// First enabled (selectable) command in the filtered list, or -1 when
	// every match is disabled.
	function firstEnabledIndex(): number {
		return filteredCommands.findIndex((c) => !c.disabled);
	}

	// Step to the next/prev enabled command, wrapping around the list.
	function stepEnabled(from: number, dir: number): number {
		const n = filteredCommands.length;
		if (n === 0) return -1;
		for (let i = 1; i <= n; i++) {
			const idx = (from + dir * i + n) % n;
			if (!filteredCommands[idx].disabled) return idx;
		}
		return -1;
	}

	$effect(() => {
		if (isOpen) {
			hoveredIndex = firstEnabledIndex();
		}
	});

	// Keep the highlight on an enabled command when the filtered list
	// changes (typing more chars, availability flipping).
	$effect(() => {
		if (hoveredIndex < 0 || hoveredIndex >= filteredCommands.length) {
			hoveredIndex = firstEnabledIndex();
			return;
		}
		if (filteredCommands[hoveredIndex].disabled) {
			hoveredIndex = firstEnabledIndex();
		}
	});

	function handleSelect(command: ChatFormCommand) {
		if (command.disabled) return;
		onSelect(command);
		onClose();
	}

	export function handleKeydown(event: KeyboardEvent): boolean {
		if (!isOpen) return false;

		if (event.key === KeyboardKey.ESCAPE) {
			event.preventDefault();
			onClose();
			return true;
		}

		if (event.key === KeyboardKey.ARROW_DOWN) {
			event.preventDefault();
			const next = hoveredIndex < 0 ? firstEnabledIndex() : stepEnabled(hoveredIndex, 1);
			if (next >= 0) {
				hoveredIndex = next;
				scrollTrigger++;
			}
			return true;
		}

		if (event.key === KeyboardKey.ARROW_UP) {
			event.preventDefault();
			const next = hoveredIndex < 0 ? firstEnabledIndex() : stepEnabled(hoveredIndex, -1);
			if (next >= 0) {
				hoveredIndex = next;
				scrollTrigger++;
			}
			return true;
		}

		if (event.key === KeyboardKey.ENTER) {
			if (hoveredIndex >= 0 && filteredCommands[hoveredIndex]) {
				event.preventDefault();
				handleSelect(filteredCommands[hoveredIndex]);
				return true;
			}
			// No selectable command - let the textarea's Enter-to-submit run.
			return false;
		}

		return false;
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
		selectedIndex={hoveredIndex}
		showSearchInput={false}
		searchQuery={query ?? ''}
		emptyMessage="No matching command"
		itemKey={(command) => command.name}
		{scrollTrigger}
	>
		{#snippet item(command, index, isSelected)}
			{@const Icon = commandIcon[command.action]}
			<ChatFormPickerListItem
				dataIndex={index}
				{isSelected}
				disabled={command.disabled}
				onclick={() => handleSelect(command)}
				onmouseenter={() => {
					if (!command.disabled) hoveredIndex = index;
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
