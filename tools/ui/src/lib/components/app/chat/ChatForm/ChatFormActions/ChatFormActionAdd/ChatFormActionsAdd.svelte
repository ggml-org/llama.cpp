<script lang="ts">
	import { isMobile } from '$lib/stores/viewport.svelte';
	import ChatFormActionAddDropdown from './ChatFormActionAddDropdown.svelte';
	import ChatFormActionAddSheet from './ChatFormActionAddSheet.svelte';
	import ChatFormActionAddButton from './ChatFormActionAddButton.svelte';
	import type { MCPPromptInfo } from '$lib/types';

	interface Props {
		disabled?: boolean;
		hasAudioModality?: boolean;
		hasVideoModality?: boolean;
		hasMcpPromptsSupport?: boolean;
		hasMcpResourcesSupport?: boolean;
		hasVisionModality?: boolean;
		onFileUpload?: () => void;
		onMcpPromptClick?: (prompt?: MCPPromptInfo) => void;
		onMcpResourcesClick?: () => void;
		onSystemPromptClick?: () => void;
		onSystemPromptWithContent?: (content: string, instructionId?: string, title?: string) => void;
	}

	let {
		disabled = false,
		hasAudioModality = false,
		hasVideoModality = false,
		hasMcpPromptsSupport = false,
		hasMcpResourcesSupport = false,
		hasVisionModality = false,
		onFileUpload,
		onMcpPromptClick,
		onMcpResourcesClick,
		onSystemPromptClick,
		onSystemPromptWithContent
	}: Props = $props();
</script>

{#if isMobile.current}
	<ChatFormActionAddSheet
		{disabled}
		{hasAudioModality}
		{hasVideoModality}
		{hasVisionModality}
		{hasMcpPromptsSupport}
		{hasMcpResourcesSupport}
		{onFileUpload}
		{onSystemPromptClick}
		{onMcpPromptClick}
		{onMcpResourcesClick}
		{onSystemPromptWithContent}
	>
		{#snippet trigger({ disabled, onclick })}
			<ChatFormActionAddButton {disabled} {onclick} />
		{/snippet}
	</ChatFormActionAddSheet>
{:else}
	<ChatFormActionAddDropdown
		{disabled}
		{hasAudioModality}
		{hasVideoModality}
		{hasVisionModality}
		{hasMcpPromptsSupport}
		{hasMcpResourcesSupport}
		{onFileUpload}
		{onMcpPromptClick}
		{onMcpResourcesClick}
		{onSystemPromptClick}
		{onSystemPromptWithContent}
	/>
{/if}
