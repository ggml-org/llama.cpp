<script lang="ts">
	import type { Snippet } from 'svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import * as Sheet from '$lib/components/ui/sheet';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import { ChevronDown, ChevronRight, File, MessageSquare, Zap, FolderOpen } from '@lucide/svelte';
	import { TOOLTIP_DELAY_DURATION } from '$lib/constants';
	import { ATTACHMENT_FILE_ITEMS } from '$lib/constants/attachment-menu';
	import { McpLogo } from '$lib/components/app';
	import { useAttachmentMenu } from '$lib/hooks/use-attachment-menu.svelte';
	import { AttachmentMenuItemId } from '$lib/enums';
	import { PencilRuler, Lightbulb } from '@lucide/svelte';
	import { ROUTES, SETTINGS_SECTION_SLUGS } from '$lib/constants/routes';
	import { RouterService } from '$lib/services/router.service';
	import { conversationsStore } from '$lib/stores/conversations.svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		hasAudioModality?: boolean;
		hasVideoModality?: boolean;
		hasVisionModality?: boolean;
		hasMcpPromptsSupport?: boolean;
		hasMcpResourcesSupport?: boolean;
		onFileUpload?: () => void;
		onSystemPromptClick?: () => void;
		onMcpPromptClick?: () => void;
		onMcpResourcesClick?: () => void;
		trigger: Snippet<[{ disabled: boolean; onclick?: () => void }]>;
	}

	let {
		class: className = '',
		disabled = false,
		hasAudioModality = false,
		hasVisionModality = false,
		hasVideoModality = false,
		hasMcpPromptsSupport = false,
		hasMcpResourcesSupport = false,
		onFileUpload,
		onSystemPromptClick,
		onMcpPromptClick,
		onMcpResourcesClick,
		trigger
	}: Props = $props();

	let sheetOpen = $state(false);
	let filesExpanded = $state(true);

	const attachmentMenu = useAttachmentMenu(
		() => ({
			hasVisionModality,
			hasAudioModality,
			hasVideoModality,
			hasMcpPromptsSupport,
			hasMcpResourcesSupport
		}),
		() => ({ onFileUpload, onSystemPromptClick, onMcpPromptClick, onMcpResourcesClick }),
		() => {
			sheetOpen = false;
		}
	);

	const sheetItemClass =
		'flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left text-sm transition-colors hover:bg-accent active:bg-accent disabled:cursor-not-allowed disabled:opacity-50';
</script>

<div class="flex items-center gap-1 {className}">
	<Sheet.Root bind:open={sheetOpen}>
		{@render trigger({ disabled, onclick: () => (sheetOpen = true) })}

		<Sheet.Content side="bottom" class="max-h-[85vh] gap-0 overflow-y-auto">
			<Sheet.Header>
				<Sheet.Title>Add to chat</Sheet.Title>

				<Sheet.Description class="sr-only">
					Add files, system prompt or configure MCP servers
				</Sheet.Description>
			</Sheet.Header>

			<div class="flex flex-col gap-1 px-1.5 pb-2">
				<Collapsible.Root open={filesExpanded} onOpenChange={(open) => (filesExpanded = open)}>
					<Collapsible.Trigger class={sheetItemClass}>
						{#if filesExpanded}
							<ChevronDown class="h-4 w-4 shrink-0" />
						{:else}
							<ChevronRight class="h-4 w-4 shrink-0" />
						{/if}

						<File class="h-4 w-4 shrink-0" />

						<span class="flex-1">Add files</span>
					</Collapsible.Trigger>

					<Collapsible.Content>
						<div class="flex flex-col gap-0.5 pl-4">
							{#each ATTACHMENT_FILE_ITEMS as item (item.id)}
								{@const enabled = attachmentMenu.isItemEnabled(item.enabledWhen)}
								{#if enabled}
									<button
										type="button"
										class={sheetItemClass}
										onclick={() => attachmentMenu.callbacks[item.action]()}
									>
										<item.icon class="h-4 w-4 shrink-0" />

										<span>{item.label}</span>
									</button>
								{:else if item.disabledTooltip}
									<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
										<Tooltip.Trigger>
											<button type="button" class={sheetItemClass} disabled>
												<item.icon class="h-4 w-4 shrink-0" />

												<span>{item.label}</span>
											</button>
										</Tooltip.Trigger>

										<Tooltip.Content side="right">
											<p>{item.disabledTooltip}</p>
										</Tooltip.Content>
									</Tooltip.Root>
								{/if}
							{/each}

							{#if !attachmentMenu.isItemEnabled('hasVisionModality')}
								{@const pdfItem = ATTACHMENT_FILE_ITEMS.find(
									(i) => i.id === AttachmentMenuItemId.PDF
								)}
								{#if pdfItem}
									<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
										<Tooltip.Trigger>
											<button
												type="button"
												class={sheetItemClass}
												onclick={() => attachmentMenu.callbacks[pdfItem.action]()}
											>
												<pdfItem.icon class="h-4 w-4 shrink-0" />

												<span>{pdfItem.label}</span>
											</button>
										</Tooltip.Trigger>

										<Tooltip.Content side="right">
											<p>PDFs will be converted to text. Image-based PDFs may not work properly.</p>
										</Tooltip.Content>
									</Tooltip.Root>
								{/if}
							{/if}
						</div>
					</Collapsible.Content>
				</Collapsible.Root>

				<button type="button" class={sheetItemClass} onclick={onSystemPromptClick}>
					<MessageSquare class="h-4 w-4 shrink-0" />

					<span>System Message</span>
				</button>

				<div class="my-2 border-t"></div>

				<button
					type="button"
					class={sheetItemClass}
					onclick={() =>
						conversationsStore.setThinkingEnabled(!conversationsStore.getThinkingEnabled())}
				>
					<Lightbulb
						class="h-4 w-4 shrink-0 {conversationsStore.getThinkingEnabled()
							? 'text-white'
							: 'text-muted-foreground'}"
					/>

					<span class="flex-1">Thinking</span>

					<span class="text-xs text-muted-foreground">
						{conversationsStore.getThinkingEnabled() ? 'on' : 'off'}
					</span>
				</button>

				<a href={ROUTES.MCP_SERVERS} class="flex items-center gap-3 px-3 py-2">
					<McpLogo class="inline h-4 w-4" />

					<span class="text-sm">MCP Servers</span>
				</a>

				<a
					href={RouterService.settings(SETTINGS_SECTION_SLUGS.TOOLS)}
					class="flex items-center gap-3 px-3 py-2"
				>
					<PencilRuler class="inline h-4 w-4" />

					<span class="text-sm">Tools</span>
				</a>

				<div class="my-2 border-t"></div>

				{#if hasMcpPromptsSupport}
					<button type="button" class={sheetItemClass} onclick={onMcpPromptClick}>
						<Zap class="h-4 w-4 shrink-0" />

						<span>MCP Prompt</span>
					</button>
				{/if}

				{#if hasMcpResourcesSupport}
					<button type="button" class={sheetItemClass} onclick={onMcpResourcesClick}>
						<FolderOpen class="h-4 w-4 shrink-0" />

						<span>MCP Resources</span>
					</button>
				{/if}
			</div>
		</Sheet.Content>
	</Sheet.Root>
</div>
