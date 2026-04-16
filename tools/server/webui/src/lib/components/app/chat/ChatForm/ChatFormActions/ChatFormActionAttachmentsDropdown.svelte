<script lang="ts">
	import { page } from '$app/state';
	import { Plus } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { TOOLTIP_DELAY_DURATION } from '$lib/constants';
	import {
		ATTACHMENT_FILE_ITEMS,
		ATTACHMENT_EXTRA_ITEMS,
		ATTACHMENT_MCP_ITEMS,
		ATTACHMENT_TOOLTIP_TEXT
	} from '$lib/constants/attachment-menu';
	import ChatFormActionToolsSubmenu from './ChatFormActionToolsSubmenu.svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		hasAudioModality?: boolean;
		hasVisionModality?: boolean;
		hasMcpPromptsSupport?: boolean;
		hasMcpResourcesSupport?: boolean;
		onFileUpload?: () => void;
		onSystemPromptClick?: () => void;
		onMcpPromptClick?: () => void;
		onMcpResourcesClick?: () => void;
	}

	let {
		class: className = '',
		disabled = false,
		hasAudioModality = false,
		hasVisionModality = false,
		hasMcpPromptsSupport = false,
		hasMcpResourcesSupport = false,
		onFileUpload,
		onSystemPromptClick,
		onMcpPromptClick,
		onMcpResourcesClick
	}: Props = $props();

	let isNewChat = $derived(!page.params.id);

	let dropdownOpen = $state(false);

	const callbacks: Record<string, (() => void) | undefined> = {
		onFileUpload: undefined,
		onSystemPromptClick: undefined,
		onMcpPromptClick: undefined,
		onMcpResourcesClick: undefined
	};

	const modalityFlags = $derived({
		hasVisionModality,
		hasAudioModality,
		hasMcpPromptsSupport,
		hasMcpResourcesSupport
	});

	function isItemEnabled(enabledWhen: string | undefined): boolean {
		if (!enabledWhen || enabledWhen === 'always') return true;
		return !!modalityFlags[enabledWhen as keyof typeof modalityFlags];
	}

	function isItemVisible(visibleWhen: string | undefined): boolean {
		if (!visibleWhen) return true;
		return !!modalityFlags[visibleWhen as keyof typeof modalityFlags];
	}

	function getSystemMessageTooltip(): string {
		return isNewChat
			? 'Add custom system message for a new conversation'
			: 'Inject custom system message at the beginning of the conversation';
	}

	$effect(() => {
		callbacks.onFileUpload = onFileUpload;
		callbacks.onSystemPromptClick = onSystemPromptClick;
		callbacks.onMcpPromptClick = () => {
			dropdownOpen = false;
			onMcpPromptClick?.();
		};
		callbacks.onMcpResourcesClick = () => {
			dropdownOpen = false;
			onMcpResourcesClick?.();
		};
	});
</script>

<div class="flex items-center gap-1 {className}">
	<DropdownMenu.Root bind:open={dropdownOpen}>
		<DropdownMenu.Trigger name="Attach files" {disabled}>
			<Tooltip.Root>
				<Tooltip.Trigger class="w-full">
					<Button
						class="file-upload-button h-8 w-8 rounded-full p-0"
						{disabled}
						variant="secondary"
						type="button"
					>
						<span class="sr-only">{ATTACHMENT_TOOLTIP_TEXT}</span>

						<Plus class="h-4 w-4" />
					</Button>
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>{ATTACHMENT_TOOLTIP_TEXT}</p>
				</Tooltip.Content>
			</Tooltip.Root>
		</DropdownMenu.Trigger>

		<DropdownMenu.Content align="start" class="w-48">
			{#each ATTACHMENT_FILE_ITEMS as item (item.id)}
				{@const enabled = isItemEnabled(item.enabledWhen)}
				{#if enabled}
					<DropdownMenu.Item
						class="{item.class ?? ''} flex cursor-pointer items-center gap-2"
						onclick={() => callbacks[item.action]?.()}
					>
						<item.icon class="h-4 w-4" />

						<span>{item.label}</span>
					</DropdownMenu.Item>
				{:else if item.disabledTooltip}
					<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
						<Tooltip.Trigger class="w-full">
							<DropdownMenu.Item
								class="{item.class ?? ''} flex cursor-pointer items-center gap-2"
								disabled
							>
								<item.icon class="h-4 w-4" />

								<span>{item.label}</span>
							</DropdownMenu.Item>
						</Tooltip.Trigger>

						<Tooltip.Content side="right">
							<p>{item.disabledTooltip}</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}
			{/each}

			{#if !isItemEnabled('hasVisionModality')}
				<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
					<Tooltip.Trigger class="w-full">
						<DropdownMenu.Item
							class="flex cursor-pointer items-center gap-2"
							onclick={() => callbacks.onFileUpload?.()}
						>
							{@const pdfItem = ATTACHMENT_FILE_ITEMS.find((i) => i.id === 'pdf')}
							{#if pdfItem}
								<pdfItem.icon class="h-4 w-4" />

								<span>{pdfItem.label}</span>
							{/if}
						</DropdownMenu.Item>
					</Tooltip.Trigger>

					<Tooltip.Content side="right">
						<p>PDFs will be converted to text. Image-based PDFs may not work properly.</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/if}

			<DropdownMenu.Separator />

			{#each ATTACHMENT_EXTRA_ITEMS as item (item.id)}
				{#if item.id === 'system-message'}
					<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
						<Tooltip.Trigger class="w-full">
							<DropdownMenu.Item
								class="flex cursor-pointer items-center gap-2"
								onclick={() => callbacks[item.action]?.()}
							>
								<item.icon class="h-4 w-4" />

								<span>{item.label}</span>
							</DropdownMenu.Item>
						</Tooltip.Trigger>

						<Tooltip.Content side="right">
							<p>{getSystemMessageTooltip()}</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}
			{/each}

			<ChatFormActionToolsSubmenu />

			{#each ATTACHMENT_MCP_ITEMS as item (item.id)}
				{#if isItemVisible(item.visibleWhen)}
					<DropdownMenu.Item
						class="flex cursor-pointer items-center gap-2"
						onclick={() => callbacks[item.action]?.()}
					>
						<item.icon class="h-4 w-4" />

						<span>{item.label}</span>
					</DropdownMenu.Item>
				{/if}
			{/each}
		</DropdownMenu.Content>
	</DropdownMenu.Root>
</div>
