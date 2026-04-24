<script lang="ts">
	import { IsMobile } from '$lib/hooks/is-mobile.svelte';
	import ChatFormActionAddDropdown from './ChatFormActionAddDropdown.svelte';
	import ChatFormActionAddSheet from './ChatFormActionAddSheet.svelte';
	import ChatFormActionAddButton from './ChatFormActionAddButton.svelte';

	interface Props {
		disabled?: boolean;
		hasAudioModality?: boolean;
		hasVisionModality?: boolean;
		hasMcpPromptsSupport?: boolean;
		hasMcpResourcesSupport?: boolean;
		onFileUpload?: () => void;
		onSystemPromptClick?: () => void;
		onMcpPromptClick?: () => void;
		onMcpSettingsClick?: () => void;
		onMcpResourcesClick?: () => void;
	}

	let {
		disabled = false,
		hasAudioModality = false,
		hasVisionModality = false,
		hasMcpPromptsSupport = false,
		hasMcpResourcesSupport = false,
		onFileUpload,
		onSystemPromptClick,
		onMcpPromptClick,
		onMcpSettingsClick,
		onMcpResourcesClick,
	}: Props = $props();

	let isMobile = new IsMobile();
</script>

{#if isMobile.current}
	<ChatFormActionAddSheet
		{disabled}
		{hasAudioModality}
		{hasVisionModality}
		{hasMcpPromptsSupport}
		{hasMcpResourcesSupport}
		{onFileUpload}
		{onSystemPromptClick}
		{onMcpPromptClick}
		{onMcpResourcesClick}
	>
		{#snippet trigger({ disabled, onclick })}
			 <ChatFormActionAddButton {disabled} {onclick} />
		{/snippet}
	</ChatFormActionAddSheet>
{:else}
	<ChatFormActionAddDropdown
		{disabled}
		{hasAudioModality}
		{hasVisionModality}
		{hasMcpPromptsSupport}
		{hasMcpResourcesSupport}
		{onFileUpload}
		{onSystemPromptClick}
		{onMcpPromptClick}
		{onMcpResourcesClick}
		onMcpSettingsClick={onMcpSettingsClick}
	>
		{#snippet trigger()}
			<ChatFormActionAddButton {disabled} />
		{/snippet}
	</ChatFormActionAddDropdown>
{/if}
