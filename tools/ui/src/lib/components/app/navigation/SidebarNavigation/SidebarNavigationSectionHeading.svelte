<script lang="ts">
	import { ChevronDown } from '@lucide/svelte';
	import { TruncatedText } from '$lib/components/app';
	import { sidebarDragStore } from '$lib/stores';
	import type { Component, Snippet } from 'svelte';

	interface Props {
		class?: string;
		/** Section id the heading drags its section by. */
		dragId: string;
		/** Keeps the heading lit, e.g. while its menu is open. */
		highlighted?: boolean;
		icon?: Component;
		label: string;
		onToggle: () => void;
		open: boolean;
		/** Sits at the far end, e.g. a "more actions" menu. */
		trailing?: Snippet;
	}

	let {
		class: className = '',
		dragId,
		highlighted = false,
		icon: Icon,
		label,
		onToggle,
		open,
		trailing
	}: Props = $props();
</script>

<!-- the whole row folds the section, all but its trailing menu; the chevron
     follows the title, not the far end, as it folds what the title names. It
     shows on hover, and stays while the section is folded so a folded
     section still reads as one -->
<!-- dragging the title moves the whole section among the others -->
<div
	class="group/heading flex h-8 min-w-0 shrink-0 items-center gap-0.5 pr-1 text-xs font-medium text-muted-foreground {highlighted
		? 'text-foreground'
		: ''} {className}"
>
	<button
		aria-expanded={open}
		class="flex h-full min-w-0 flex-1 cursor-pointer items-center gap-1 rounded-md pl-2 text-left transition-colors hover:text-foreground"
		onclick={onToggle}
		onpointerdown={(event) => sidebarDragStore.pressSection(event, dragId)}
		type="button"
	>
		{#if Icon}
			<Icon class="h-3.5 w-3.5 shrink-0" />
		{/if}

		<TruncatedText showTooltip={false} text={label} />

		<ChevronDown
			aria-hidden="true"
			class="h-3.5 w-3.5 shrink-0 transition-[opacity,rotate] duration-200 ease-out group-hover/heading:opacity-100 [button:focus-visible_&]:opacity-100 {open
				? 'opacity-0'
				: '-rotate-90 opacity-100'}"
		/>
	</button>

	{#if trailing}
		<div class="shrink-0">{@render trailing()}</div>
	{/if}
</div>
