<script lang="ts">
	import { Eye } from '@lucide/svelte';
	import { ActionIcon, ActionIconCopyToClipboard } from '$lib/components/app';
	import { FileTypeText } from '$lib/enums';

	interface Props {
		code: string;
		language: string;
		disabled?: boolean;
		onPreview?: (code: string, language: string) => void;
	}

	let { code, language, disabled = false, onPreview }: Props = $props();

	const showPreview = $derived(language?.toLowerCase() === FileTypeText.HTML);
</script>

<div class="code-block-actions">
	<ActionIconCopyToClipboard
		text={code}
		canCopy={!disabled}
		ariaLabel={disabled ? 'Code incomplete' : 'Copy code'}
	/>

	{#if showPreview}
		<ActionIcon
			icon={Eye}
			tooltip={disabled ? 'Code incomplete' : 'Preview code'}
			{disabled}
			onclick={() => onPreview!(code, language)}
		/>
	{/if}
</div>
