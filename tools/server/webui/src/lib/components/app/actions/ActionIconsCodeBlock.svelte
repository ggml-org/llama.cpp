<script lang="ts">
	import { Copy, Eye } from '@lucide/svelte';
	import { copyCodeToClipboard } from '$lib/utils';
	import { FileTypeText } from '$lib/enums';

	interface Props {
		code: string;
		language: string;
		disabled?: boolean;
		onPreview?: (code: string, language: string) => void;
	}

	let { code, language, disabled = false, onPreview }: Props = $props();

	const showPreview = $derived(language?.toLowerCase() === FileTypeText.HTML);

	function handleCopy() {
		if (disabled) return;
		copyCodeToClipboard(code);
	}

	function handlePreview() {
		if (disabled) return;
		onPreview?.(code, language);
	}
</script>

<div class="code-block-actions">
	<button
		class="copy-code-btn"
		class:opacity-50={disabled}
		class:!cursor-not-allowed={disabled}
		title={disabled ? 'Code incomplete' : 'Copy code'}
		aria-label="Copy code"
		aria-disabled={disabled}
		type="button"
		onclick={handleCopy}
	>
		<Copy size={16} />
	</button>
	{#if showPreview}
		<button
			class="preview-code-btn"
			class:opacity-50={disabled}
			class:!cursor-not-allowed={disabled}
			title={disabled ? 'Code incomplete' : 'Preview code'}
			aria-label="Preview code"
			aria-disabled={disabled}
			type="button"
			onclick={handlePreview}
		>
			<Eye size={16} />
		</button>
	{/if}
</div>
