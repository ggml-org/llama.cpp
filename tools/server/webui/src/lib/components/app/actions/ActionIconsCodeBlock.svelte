<script lang="ts">
	import { Eye } from '@lucide/svelte';
	import { ActionIconCopyToClipboard } from '$lib/components/app';
	import { FileTypeText } from '$lib/enums';

	interface Props {
		code: string;
		language: string;
		disabled?: boolean;
		onPreview?: (code: string, language: string) => void;
	}

	let { code, language, disabled = false, onPreview }: Props = $props();

	const showPreview = $derived(language?.toLowerCase() === FileTypeText.HTML);

	function handlePreview() {
		if (disabled) return;
		onPreview?.(code, language);
	}
</script>

<div class="code-block-actions">
	<div class="copy-code-btn" class:opacity-50={disabled} class:!cursor-not-allowed={disabled}>
		<ActionIconCopyToClipboard
			text={code}
			canCopy={!disabled}
			ariaLabel={disabled ? 'Code incomplete' : 'Copy code'}
		/>
	</div>

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
