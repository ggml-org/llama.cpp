<script lang="ts">
	import type { Snippet } from 'svelte';
	import * as Popover from '$lib/components/ui/popover';

	interface Props {
		class?: string;
		isOpen?: boolean;
		srLabel?: string;
		onClose?: () => void;
		onKeydown?: (event: KeyboardEvent) => void;
		children: Snippet;
	}

	let {
		class: className = '',
		isOpen = $bindable(false),
		srLabel = 'Open picker',
		onClose,
		onKeydown,
		children
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
	<Popover.Trigger class="pointer-events-none absolute inset-0 opacity-0">
		<span class="sr-only">{srLabel}</span>
	</Popover.Trigger>

	<Popover.Content
		side="top"
		align="start"
		sideOffset={12}
		class="w-[var(--bits-popover-anchor-width)] max-w-none rounded-xl border-border/50 p-0 shadow-xl {className}"
		onkeydown={onKeydown}
		onOpenAutoFocus={(e) => e.preventDefault()}
	>
		{@render children()}
	</Popover.Content>
</Popover.Root>
