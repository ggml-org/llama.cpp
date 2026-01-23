<script lang="ts">
	import { t } from '$lib/i18n';

	interface Props {
		class?: string;
		show?: boolean;
	}

	let { class: className = '', show = true }: Props = $props();
</script>

{#if show}
	<div class="mt-4 flex items-center justify-center {className}">
		<p class="text-xs text-muted-foreground">
			{t('chat.form.helper.press')}{' '}
			<kbd class="rounded bg-muted px-1 py-0.5 font-mono text-xs">
				{t('chat.form.helper.enter')}
			</kbd>
			{' '}
			{t('chat.form.helper.to_send')},{' '}
			<kbd class="rounded bg-muted px-1 py-0.5 font-mono text-xs">
				{t('chat.form.helper.shift_enter')}
			</kbd>
			{' '}
			{t('chat.form.helper.for_new_line')}
		</p>
	</div>
{/if}
