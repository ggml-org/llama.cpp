<script lang="ts">
	import { Plus, MessageSquare, Zap } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { FILE_TYPE_ICONS } from '$lib/constants/icons';
	import { McpLogo } from '$lib/components/app';

	interface Props {
		class?: string;
		disabled?: boolean;
		hasAudioModality?: boolean;
		hasVisionModality?: boolean;
		hasMcpPromptsSupport?: boolean;
		onFileUpload?: () => void;
		onSystemPromptClick?: () => void;
		onMcpPromptClick?: () => void;
		onMcpServersClick?: () => void;
	}

	let {
		class: className = '',
		disabled = false,
		hasAudioModality = false,
		hasVisionModality = false,
		hasMcpPromptsSupport = false,
		onFileUpload,
		onSystemPromptClick,
		onMcpPromptClick,
		onMcpServersClick
	}: Props = $props();

	let dropdownOpen = $state(false);

	function handleMcpPromptClick() {
		dropdownOpen = false;
		onMcpPromptClick?.();
	}

	function handleMcpServersClick() {
		dropdownOpen = false;
		onMcpServersClick?.();
	}

	const fileUploadTooltipText = 'Add files, system prompt or MCP Servers';
</script>

<div class="flex items-center gap-1 {className}">
	<DropdownMenu.Root bind:open={dropdownOpen}>
		<DropdownMenu.Trigger name="Attach files" {disabled}>
			<Tooltip.Root>
				<Tooltip.Trigger>
					<Button
						class="file-upload-button h-8 w-8 rounded-full p-0"
						{disabled}
						variant="secondary"
						type="button"
					>
						<span class="sr-only">{fileUploadTooltipText}</span>

						<Plus class="h-4 w-4" />
					</Button>
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>{fileUploadTooltipText}</p>
				</Tooltip.Content>
			</Tooltip.Root>
		</DropdownMenu.Trigger>

		<DropdownMenu.Content align="start" class="w-48">
			<Tooltip.Root>
				<Tooltip.Trigger class="w-full">
					<DropdownMenu.Item
						class="images-button flex cursor-pointer items-center gap-2"
						disabled={!hasVisionModality}
						onclick={() => onFileUpload?.()}
					>
						<FILE_TYPE_ICONS.image class="h-4 w-4" />

						<span>Images</span>
					</DropdownMenu.Item>
				</Tooltip.Trigger>

				{#if !hasVisionModality}
					<Tooltip.Content usePortal={false}>
						<p>Images require vision models to be processed</p>
					</Tooltip.Content>
				{/if}
			</Tooltip.Root>

			<Tooltip.Root>
				<Tooltip.Trigger class="w-full">
					<DropdownMenu.Item
						class="audio-button flex cursor-pointer items-center gap-2"
						disabled={!hasAudioModality}
						onclick={() => onFileUpload?.()}
					>
						<FILE_TYPE_ICONS.audio class="h-4 w-4" />

						<span>Audio Files</span>
					</DropdownMenu.Item>
				</Tooltip.Trigger>

				{#if !hasAudioModality}
					<Tooltip.Content usePortal={false}>
						<p>Audio files require audio models to be processed</p>
					</Tooltip.Content>
				{/if}
			</Tooltip.Root>

			<DropdownMenu.Item
				class="flex cursor-pointer items-center gap-2"
				onclick={() => onFileUpload?.()}
			>
				<FILE_TYPE_ICONS.text class="h-4 w-4" />

				<span>Text Files</span>
			</DropdownMenu.Item>

			<Tooltip.Root>
				<Tooltip.Trigger class="w-full">
					<DropdownMenu.Item
						class="flex cursor-pointer items-center gap-2"
						onclick={() => onFileUpload?.()}
					>
						<FILE_TYPE_ICONS.pdf class="h-4 w-4" />

						<span>PDF Files</span>
					</DropdownMenu.Item>
				</Tooltip.Trigger>

				{#if !hasVisionModality}
					<Tooltip.Content usePortal={false}>
						<p>PDFs will be converted to text. Image-based PDFs may not work properly.</p>
					</Tooltip.Content>
				{/if}
			</Tooltip.Root>

			<DropdownMenu.Separator />
			<Tooltip.Root>
				<Tooltip.Trigger class="w-full">
					<DropdownMenu.Item
						class="flex cursor-pointer items-center gap-2"
						onclick={() => onSystemPromptClick?.()}
					>
						<MessageSquare class="h-4 w-4" />

						<span>System Message</span>
					</DropdownMenu.Item>
				</Tooltip.Trigger>

				<Tooltip.Content usePortal={false}>
					<p>Add a custom system message for this conversation</p>
				</Tooltip.Content>
			</Tooltip.Root>

			{#if hasMcpPromptsSupport}
				<Tooltip.Root>
					<Tooltip.Trigger class="w-full">
						<DropdownMenu.Item
							class="flex cursor-pointer items-center gap-2"
							onclick={handleMcpPromptClick}
						>
							<Zap class="h-4 w-4" />

							<span>MCP Prompt</span>
						</DropdownMenu.Item>
					</Tooltip.Trigger>

					<Tooltip.Content usePortal={false}>
						<p>Insert a prompt from an MCP server</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/if}

			<DropdownMenu.Separator />
			<Tooltip.Root>
				<Tooltip.Trigger class="w-full">
					<DropdownMenu.Item
						class="flex cursor-pointer items-center gap-2"
						onclick={handleMcpServersClick}
					>
						<McpLogo class="h-4 w-4" />

						<span>MCP Servers</span>
					</DropdownMenu.Item>
				</Tooltip.Trigger>

				<Tooltip.Content usePortal={false}>
					<p>Configure MCP servers for agentic tool execution</p>
				</Tooltip.Content>
			</Tooltip.Root>
		</DropdownMenu.Content>
	</DropdownMenu.Root>
</div>
