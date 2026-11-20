<script lang="ts">
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import { Plus } from '@lucide/svelte';
	import { ATTACHMENT_TOOLTIP_TEXT, ICON_CLASS_DEFAULT } from '#lib/constants/index.ts';

	interface Props {
		disabled?: boolean;
		onclick?: (e: MouseEvent) => void;
	}

	let { disabled = false, onclick }: Props = $props();
</script>

<Tooltip.Root>
	<Tooltip.Trigger class="w-full">
		<Button
			class="file-upload-button bg-background size-9 rounded-full p-0 shadow-(--elevation-key) md:size-8 dark:bg-muted-foreground/15"
			{disabled}
			{onclick}
			type="button"
			variant="ghost"
		>
			<span class="sr-only">{ATTACHMENT_TOOLTIP_TEXT}</span>

			<Plus class={ICON_CLASS_DEFAULT} />
		</Button>
	</Tooltip.Trigger>

	<Tooltip.Content>
		<p>{ATTACHMENT_TOOLTIP_TEXT}</p>
	</Tooltip.Content>
</Tooltip.Root>
