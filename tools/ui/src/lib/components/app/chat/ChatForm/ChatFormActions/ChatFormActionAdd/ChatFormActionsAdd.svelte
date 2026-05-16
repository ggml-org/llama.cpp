<script lang="ts">
	import { IsMobile } from '$lib/hooks/is-mobile.svelte';
	import ChatFormActionAddDropdown from './ChatFormActionAddDropdown.svelte';
	import ChatFormActionAddSheet from './ChatFormActionAddSheet.svelte';
	import ChatFormActionAddButton from './ChatFormActionAddButton.svelte';

	interface Props {
		disabled?: boolean;
		hasAudioModality?: boolean;
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
		hasVisionModality = false,
		onFileUpload,
		onMcpPromptClick,
		onMcpResourcesClick,
		onMcpSettingsClick,
		onSystemPromptClick
	}: Props = $props();

	const isMobile = new IsMobile();
</script>

{#if isMobile.current}
	<ChatFormActionAddSheet
		{disabled}
		{hasAudioModality}
		{hasVisionModality}
		{hasMcpPromptsSupport}
		{hasMcpResourcesSupport}
		{onFileUpload}
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
		{onMcpPromptClick}
		{onMcpResourcesClick}
		{onMcpSettingsClick}
		{onSystemPromptClick}
	/>
{/if}
