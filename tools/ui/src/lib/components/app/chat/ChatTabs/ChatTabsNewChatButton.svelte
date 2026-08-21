<script lang="ts">
	import { Plus } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';

	interface Props {
		onclick?: () => void;
	}

	let { onclick }: Props = $props();
</script>

<Tooltip.Root>
	<Tooltip.Trigger>
		{#snippet child({ props })}
			<button
				{...props}
				class="backdrop-blur-lg flex h-8 w-8 mr-4 shrink-0 cursor-pointer items-center justify-center rounded-md transition-colors hover:bg-foreground/5"
				{onclick}
				aria-label="New chat"
			>
				<Plus class="{ICON_CLASS_DEFAULT} opacity-40 transition-opacity group-hover:opacity-100" />
			</button>
		{/snippet}
	</Tooltip.Trigger>

	<Tooltip.Content>
		<p>New chat</p>
	</Tooltip.Content>
</Tooltip.Root>
