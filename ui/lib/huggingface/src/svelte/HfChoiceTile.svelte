<script lang="ts">
	import { Check } from '@lucide/svelte';
	import type { Component, Snippet } from 'svelte';

	interface Props {
		/** The line under the name. */
		children: Snippet;
		/** Before the name; the picked tile shows a check in its place. */
		icon: Component<{ class?: string }>;
		isPicked: boolean;
		label: string;
		disabled?: boolean;
		onpick: () => void;
		/** The whole name, for a tile that has to cut it short. */
		title?: string;
	}

	let { children, disabled = false, icon: Icon, isPicked, label, onpick, title }: Props = $props();
</script>

<!-- one of a few choices, shown as tiles: the picked one takes the row surface and a check -->
<button
	aria-pressed={isPicked}
	class={[
		'relative flex min-w-0 cursor-pointer flex-col items-start gap-1 rounded-xl border px-3 py-2.5 text-left outline-none',
		'transition-colors duration-200 ease-out-quint focus-visible:ring-3 focus-visible:ring-ring/50',
		'disabled:pointer-events-none disabled:opacity-50',
		isPicked
			? 'selected-row-surface border-transparent'
			: 'matte-key border-(--line-control) bg-(--field-background) hover:bg-(--action-ghost-hover)'
	]}
	{disabled}
	onclick={onpick}
	{title}
	type="button"
>
	<span class="flex w-full min-w-0 items-center gap-1.5 text-label">
		{#if isPicked}
			<Check class="size-icon-sm shrink-0 text-(--action-primary)" />
		{:else}
			<Icon class="size-icon-sm shrink-0 text-muted-foreground" />
		{/if}

		<span class="truncate">{label}</span>
	</span>

	<span class="flex w-full min-w-0 items-center gap-1 text-caption text-muted-foreground">
		{@render children()}
	</span>
</button>
