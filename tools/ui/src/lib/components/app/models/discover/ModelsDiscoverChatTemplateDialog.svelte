<script lang="ts">
	import { Copy } from '@lucide/svelte';
	import * as Dialog from '$lib/components/ui/dialog';
	import { copyToClipboard } from '$lib/utils';

	interface Props {
		open?: boolean;
		chatTemplate: string;
		onOpenChange?: (open: boolean) => void;
	}

	let { chatTemplate, onOpenChange, open = $bindable(false) }: Props = $props();

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content
		class="flex max-h-[calc(100vh-4rem)] flex-col gap-0 p-0 md:w-[calc(100vw-4rem)]! md:max-w-4xl!"
	>
		<Dialog.Header class="flex-row items-center justify-between border-b border-border/40 p-4">
			<Dialog.Title class="text-sm font-semibold">Chat template</Dialog.Title>
			<button
				type="button"
				onclick={() => copyToClipboard(chatTemplate)}
				class="inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-medium transition-colors hover:bg-muted"
			>
				<Copy class="h-3.5 w-3.5" />
				Copy
			</button>
		</Dialog.Header>

		<pre
			class="flex-1 overflow-auto p-4 font-mono text-xs break-all whitespace-pre-wrap text-muted-foreground">{chatTemplate}</pre>
	</Dialog.Content>
</Dialog.Root>
