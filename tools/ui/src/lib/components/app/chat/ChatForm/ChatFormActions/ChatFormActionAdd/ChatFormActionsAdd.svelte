<script lang="ts">
	import ChatFormActionAddButton from './ChatFormActionAddButton.svelte';
	import ChatFormActionAddDropdown from './ChatFormActionAddDropdown.svelte';
	import ChatFormActionAddSheet from './ChatFormActionAddSheet.svelte';
	import { isMobile } from '$lib/stores/viewport.svelte';

	interface Props {
		disabled?: boolean;
		hasAudioModality?: boolean;
		hasVideoModality?: boolean;
		hasMcpPromptsSupport?: boolean;
		hasMcpResourcesSupport?: boolean;
		hasVisionModality?: boolean;
		onFileUpload?: () => void;
		onMcpPromptClick?: () => void;
		onMcpResourcesClick?: () => void;
		onMcpSettingsClick?: () => void;
		onSystemPromptClick?: () => void;
	}

	let {
		disabled = false,
		hasAudioModality = false,
		hasMcpPromptsSupport = false,
		hasMcpResourcesSupport = false,
		hasVideoModality = false,
		hasVisionModality = false,
		onFileUpload,
		onMcpPromptClick,
		onMcpResourcesClick,
		onMcpSettingsClick,
		onSystemPromptClick
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
		{onMcpSettingsClick}
		{onSystemPromptClick}
	/>
{/if}
