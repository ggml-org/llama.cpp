<script lang="ts">
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { FileX } from '@lucide/svelte';

	interface Props {
		open: boolean;
		emptyFiles: string[];
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), emptyFiles, onOpenChange }: Props = $props();

	function handleOpenChange(newOpen: boolean) {
		open = newOpen;
		onOpenChange?.(newOpen);
	}
</script>

<AlertDialog.Root {open} onOpenChange={handleOpenChange}>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<FileX class="text-destructive h-5 w-5" />
				Empty Files Detected
			</AlertDialog.Title>
			<AlertDialog.Description>
				The following files are empty and have been removed from your attachments:
			</AlertDialog.Description>
		</AlertDialog.Header>

		<div class="space-y-3 text-sm">
			<div class="bg-muted rounded-lg p-3">
				<div class="mb-2 font-medium">Empty Files:</div>
				<ul class="text-muted-foreground list-inside list-disc space-y-1">
					{#each emptyFiles as fileName}
						<li class="font-mono text-sm">{fileName}</li>
					{/each}
				</ul>
			</div>

			<div>
				<div class="mb-2 font-medium">What happened:</div>
				<ul class="text-muted-foreground list-inside list-disc space-y-1">
					<li>Empty files cannot be processed or sent to the AI model</li>
					<li>These files have been automatically removed from your attachments</li>
					<li>You can try uploading files with content instead</li>
				</ul>
			</div>
		</div>

		<AlertDialog.Footer>
			<AlertDialog.Action onclick={() => handleOpenChange(false)}>Got it</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
