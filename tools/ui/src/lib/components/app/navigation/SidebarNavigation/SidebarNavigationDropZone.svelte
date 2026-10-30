<script lang="ts">
	import { SIDEBAR_DRAG_HIGHLIGHT_CLASS } from '$lib/constants';
	import { sidebarDragStore } from '$lib/stores';
	import type { Snippet } from 'svelte';
	import type { ClassValue } from 'svelte/elements';

	interface Props {
		class?: ClassValue;
		/** Receives the drag id of the chat or project dropped on the zone. */
		onDrop: (dragId: string) => Promise<void> | void;
		/** The section, or project, this zone is. */
		sectionId: string;
		/** Whether the chat or project is here already, or cannot go here. */
		holds: (dragId: string) => boolean;
		heading: Snippet;
		children: Snippet;
	}

	let { children, class: className, heading, holds, onDrop, sectionId }: Props = $props();

	$effect(() => sidebarDragStore.registerZone(sectionId, { holds, onDrop }));

	let isOver = $derived(sidebarDragStore.targetSectionId === sectionId);
	// the section being moved follows the pointer, marked like a section a chat is dragged over
	let isLifted = $derived(sidebarDragStore.sectionId === sectionId);
</script>

<div
	class={[
		'rounded-lg transition-[background-color,box-shadow] duration-200 ease-out motion-reduce:transition-none',
		className,
		(isOver || isLifted) && SIDEBAR_DRAG_HIGHLIGHT_CLASS
	]}
	data-sidebar-section={sectionId}
>
	{@render heading()}

	{@render children()}
</div>
