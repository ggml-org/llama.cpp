<script lang="ts">
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { Button } from '$lib/components/ui/button';
	import { t } from '$lib/i18n';

	interface Props {
		open: boolean;
		currentTitle: string;
		newTitle: string;
		onConfirm: () => void;
		onCancel: () => void;
	}

	let { open = $bindable(), currentTitle, newTitle, onConfirm, onCancel }: Props = $props();
</script>

<AlertDialog.Root bind:open>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title>{t('dialog.title_update.title')}</AlertDialog.Title>

			<AlertDialog.Description>
				{t('dialog.title_update.description')}
			</AlertDialog.Description>
		</AlertDialog.Header>

		<div class="space-y-4 pt-2 pb-6">
			<div class="space-y-2">
				<p class="text-sm font-medium text-muted-foreground">
					{t('dialog.title_update.current_label')}
				</p>

				<p class="rounded-md bg-muted/50 p-3 text-sm font-medium">{currentTitle}</p>
			</div>

			<div class="space-y-2">
				<p class="text-sm font-medium text-muted-foreground">
					{t('dialog.title_update.new_label')}
				</p>

				<p class="rounded-md bg-muted/50 p-3 text-sm font-medium">{newTitle}</p>
			</div>
		</div>

		<AlertDialog.Footer>
			<Button variant="outline" onclick={onCancel}>
				{t('dialog.title_update.keep')}
			</Button>

			<Button onclick={onConfirm}>{t('dialog.title_update.confirm')}</Button>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
