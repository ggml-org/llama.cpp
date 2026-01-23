<script>
	import { Upload } from '@lucide/svelte';
	import { t } from '$lib/i18n';
</script>

<div
	class="pointer-events-none fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
>
	<div
		class="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-border bg-background p-12 shadow-lg"
	>
		<Upload class="mb-4 h-12 w-12 text-muted-foreground" />

		<p class="text-lg font-medium text-foreground">{t('chat.screen.drag.title')}</p>

		<p class="text-sm text-muted-foreground">{t('chat.screen.drag.description')}</p>
	</div>
</div>
