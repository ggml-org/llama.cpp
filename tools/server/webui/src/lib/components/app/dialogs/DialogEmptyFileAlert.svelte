<script lang="ts">
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { FileX } from '@lucide/svelte';
	import { t } from '$lib/i18n';

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
				<FileX class="h-5 w-5 text-destructive" />

				{t('dialog.empty_files.title')}
			</AlertDialog.Title>

			<AlertDialog.Description>
				{t('dialog.empty_files.description')}
			</AlertDialog.Description>
		</AlertDialog.Header>

		<div class="space-y-3 text-sm">
			<div class="rounded-lg bg-muted p-3">
				<div class="mb-2 font-medium">{t('dialog.empty_files.list_title')}</div>

				<ul class="list-inside list-disc space-y-1 text-muted-foreground">
					{#each emptyFiles as fileName (fileName)}
						<li class="font-mono text-sm">{fileName}</li>
					{/each}
				</ul>
			</div>

			<div>
				<div class="mb-2 font-medium">{t('dialog.empty_files.what_happened_title')}</div>

				<ul class="list-inside list-disc space-y-1 text-muted-foreground">
					<li>{t('dialog.empty_files.reason_one')}</li>

					<li>{t('dialog.empty_files.reason_two')}</li>

					<li>{t('dialog.empty_files.reason_three')}</li>
				</ul>
			</div>
		</div>

		<AlertDialog.Footer>
			<AlertDialog.Action onclick={() => handleOpenChange(false)}>
				{t('dialog.empty_files.confirm')}
			</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
