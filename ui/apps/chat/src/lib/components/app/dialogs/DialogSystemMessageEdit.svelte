<script lang="ts">
	import * as AlertDialog from '@ggml-org/llama-ui/ui/alert-dialog';
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { Textarea } from '@ggml-org/llama-ui/ui/textarea';
	import { MessageSquareText } from '@lucide/svelte';

	interface Props {
		open: boolean;
		content: string;
		onSave: (content: string) => void;
	}

	let { content, onSave, open = $bindable() }: Props = $props();

	let draft = $state('');
	let textareaRef = $state<HTMLTextAreaElement | null>(null);

	const canSave = $derived(draft.trim().length > 0 && draft.trim() !== content.trim());

	$effect(() => {
		if (!open) return;

		draft = content;
		queueMicrotask(() => textareaRef?.focus());
	});

	function handleSave() {
		if (!canSave) return;

		onSave(draft.trim());
		open = false;
	}
</script>

<AlertDialog.Root bind:open>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<MessageSquareText class="size-icon-lg" />
				Custom system message
			</AlertDialog.Title>

			<AlertDialog.Description>
				The starting message that defines how the model behaves in this conversation.
			</AlertDialog.Description>
		</AlertDialog.Header>

		<Textarea
			bind:ref={textareaRef}
			bind:value={draft}
			aria-label="System message"
			class="max-h-80 min-h-32"
			onkeydown={(event) => {
				if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) handleSave();
			}}
			placeholder="System message..."
		/>

		<AlertDialog.Footer>
			<AlertDialog.Cancel>Cancel</AlertDialog.Cancel>

			<Button disabled={!canSave} onclick={handleSave} type="button">Save</Button>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
