<script lang="ts">
	import * as Popover from '$lib/components/ui/popover';
	import type { Snippet } from 'svelte';

	interface Props {
		class?: string;
		/** What the popover floats by, instead of the box its trigger fills. */
		customAnchor?: HTMLElement | null;
		isOpen?: boolean;
		srLabel?: string;
		onClose?: () => void;
		onKeydown?: (event: KeyboardEvent) => void;
		children: Snippet;
	}

	let {
		children,
		class: className = '',
		customAnchor = null,
		isOpen = $bindable(false),
		onClose,
		onKeydown,
		srLabel = 'Open picker'
	}: Props = $props();
</script>

<Popover.Root
	bind:open={isOpen}
	onOpenChange={(open) => {
		if (!open) {
			onClose?.();
		}
	}}
>
	<Popover.Trigger
		aria-hidden="true"
		class="pointer-events-none absolute inset-0 opacity-0"
		tabindex={-1}
	>
		<span class="sr-only">{srLabel}</span>
	</Popover.Trigger>

	<Popover.Content
		align="start"
		class="w-[var(--bits-popover-anchor-width)] max-w-none rounded-xl border-border/50 p-0 shadow-xl {className}"
		{customAnchor}
		onOpenAutoFocus={(event) => event.preventDefault()}
		onkeydown={onKeydown}
		preventScroll={false}
		side="top"
		sideOffset={12}
	>
		{@render children()}
	</Popover.Content>
</Popover.Root>
