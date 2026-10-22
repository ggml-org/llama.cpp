<script lang="ts">
	import { Plus } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';

	interface Props {
		label?: string;
		onclick?: () => void;
	}

	let { label = 'New chat', onclick }: Props = $props();
</script>

<Tooltip.Root>
	<Tooltip.Trigger>
		{#snippet child({ props })}
			<button
				{...props}
				aria-label={label}
				class="backdrop-blur-lg flex size-8 mr-4 shrink-0 cursor-pointer items-center justify-center rounded-md transition-colors hover:bg-foreground/5"
				{onclick}
			>
				<Plus class="{ICON_CLASS_DEFAULT} opacity-40 transition-opacity group-hover:opacity-100" />
			</button>
		{/snippet}
	</Tooltip.Trigger>

	<Tooltip.Content>
		<p>{label}</p>
	</Tooltip.Content>
</Tooltip.Root>
