<script lang="ts">
	import { Plus } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ATTACHMENT_TOOLTIP_TEXT } from '$lib/constants';

	interface Props {
		disabled?: boolean;
		onclick?: (e: MouseEvent) => void;
	}

	let { disabled = false, onclick }: Props = $props();
</script>

<Tooltip.Root>
	<Tooltip.Trigger class="w-full">
		<Button
			class="file-upload-button h-8 w-8 rounded-full p-0"
			{disabled}
			{onclick}
			variant="secondary"
			type="button"
		>
			<span class="sr-only">{ATTACHMENT_TOOLTIP_TEXT}</span>

			<Plus class="h-4 w-4" />
		</Button>
	</Tooltip.Trigger>

	<Tooltip.Content>
		<p>{ATTACHMENT_TOOLTIP_TEXT}</p>
	</Tooltip.Content>
</Tooltip.Root>
