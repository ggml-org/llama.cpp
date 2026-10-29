<script lang="ts">
	import { Check, Copy, X } from '@lucide/svelte';
	import { SyntaxHighlightedCode } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { COPY_FEEDBACK_RESET_MS } from '$lib/constants';
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

	let copied = $state(false);

	async function copy() {
		await copyToClipboard(chatTemplate);
		copied = true;
		setTimeout(() => (copied = false), COPY_FEEDBACK_RESET_MS);
	}
</script>

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content
		class="flex max-h-[calc(100vh-4rem)] flex-col gap-0 p-0 md:w-[calc(100vw-4rem)]! md:max-w-4xl!"
	>
		<!-- The header's corner-pinned close X never lines up with a padded flex row,
		     so it is replaced by one inside the row, aligned with the title -->
		<Dialog.Header
			class="flex-row items-center gap-2 border-b border-border/40 p-4"
			showCloseButton={false}
		>
			<Dialog.Title class="text-sm font-semibold">Chat template</Dialog.Title>

			<Button
				aria-label="Copy chat template"
				class="h-auto gap-1.5 px-2 py-1 text-xs"
				onclick={() => void copy()}
				size="sm"
				variant="secondary"
			>
				{#if copied}
					<Check class="size-3.5 text-green-500" />
				{:else}
					<Copy class="size-3.5" />
				{/if}

				Copy
			</Button>

			<Dialog.Close
				aria-label="Close"
				class="ml-auto inline-flex cursor-pointer items-center justify-center rounded-md p-1.5 text-muted-foreground/70 transition-colors hover:bg-muted-foreground/10 hover:text-foreground"
				type="button"
			>
				<X class="h-4 w-4" />
			</Dialog.Close>
		</Dialog.Header>

		<SyntaxHighlightedCode
			class="h-full rounded-none border-0 shadow-none"
			code={chatTemplate}
			language="jinja"
			maxHeight="100%"
		/>
	</Dialog.Content>
</Dialog.Root>
