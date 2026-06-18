<script lang="ts">
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { MarkdownContent } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import Input from '$lib/components/ui/input/input.svelte';
	import Label from '$lib/components/ui/label/label.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { Check, MoreHorizontal, Trash2, Edit, ArrowRight } from '@lucide/svelte';

	interface Props {
		id: string;
		title: string;
		content: string;
		lastModified: number;
		onDelete?: () => void;
		onEdit?: (id: string) => void;
		onStartNewChat?: (id: string, title: string, content: string) => void;
	}

	let {
		id,
		title,
		content,
		lastModified,
		onDelete,
		onEdit,
		onStartNewChat
	}: Props = $props();

	// Edit state
	let isEditing = $state(false);
	let editTitle = $state(title);
	let editContent = $state(content);
	let editTitleError = $derived.by(() => (!editTitle.trim() ? 'Title is required' : null));

	// Display state
	let isExpanded = $state(false);
	let contentElement: HTMLElement | undefined = $state();
	let contentHeight = $state(0);

	const MAX_HEIGHT = 200;
	const currentConfig = config();

	let showExpandButton = $derived(contentHeight > MAX_HEIGHT);

	$effect(() => {
		if (!contentElement || !content.trim()) return;

		const resizeObserver = new ResizeObserver((entries) => {
			for (const entry of entries) {
				const element = entry.target as HTMLElement;
				contentHeight = element.scrollHeight;
			}
		});

		resizeObserver.observe(contentElement);

		return () => {
			resizeObserver.disconnect();
		};
	});

	// Sync edit values when title/content change externally
	$effect(() => {
		if (!isEditing) {
			editTitle = title;
			editContent = content;
		}
	});

	function toggleExpand() {
		isExpanded = !isExpanded;
	}

	function handleSave() {
		onEdit?.(id);
	}

	function handleCancel() {
		isEditing = false;
	}

	function handleDelete() {
		onDelete?.();
	}

	function handleStartNewChat() {
		onStartNewChat?.(id, title, content);
	}

	let formattedDate = $derived.by(() => {
		const date = new Date(lastModified);
		return date.toLocaleDateString('en-US', {
			year: 'numeric',
			month: 'short',
			day: 'numeric'
		});
	});
</script>

<!-- Reusable more menu icon with tooltip for dropdown triggers -->
<svelte:options />

<div class="group relative flex flex-col gap-2">
	{#if !isEditing}
		{#if content.trim()}
			<div class="relative">
				<button
					class="group/expand w-full text-left {!isExpanded && showExpandButton
						? 'cursor-pointer'
						: 'cursor-auto'}"
					onclick={showExpandButton && !isExpanded ? toggleExpand : undefined}
					type="button"
				>
					<div
						class="relative overflow-hidden rounded-[1.125rem]"
						style="border: 2px dashed hsl(var(--border)); max-height: var(--max-message-height);"
					>
						<div
							class="relative flex flex-col justify-between rounded-[1.125rem] border-2 border-dashed border-border/50 bg-muted px-6 py-4"
							style="overflow-wrap: anywhere; word-break: break-word; min-height: 6rem;"
						>
							<!-- Top bar: title, date, more menu -->
							<div class="mb-2 flex items-start justify-between gap-2">
								<div class="min-w-0">
									<h3 class="text-xl font-semibold tracking-tight text-foreground">{title}</h3>
									<p class="mt-0.5 text-xs text-muted-foreground">{formattedDate}</p>
								</div>
								<DropdownMenu.Root>
									<DropdownMenu.Trigger>
										<Tooltip.Root>
											<Tooltip.Trigger>
												<Button
													variant="ghost"
													size="sm"
													class="h-6 w-6 p-0 hover:bg-transparent data-[state=open]:bg-transparent!"
													aria-label="More actions"
												>
													<MoreHorizontal class="h-3 w-3" />
												</Button>
											</Tooltip.Trigger>
											<Tooltip.Content side="bottom">
												<p>More actions</p>
											</Tooltip.Content>
										</Tooltip.Root>
									</DropdownMenu.Trigger>
									<DropdownMenu.Content class="w-32">
										<DropdownMenu.Item
											class="flex cursor-pointer items-center gap-2"
											onclick={() => onEdit?.(id)}
										>
											<Edit class="h-4 w-4" />
											<span>Edit</span>
										</DropdownMenu.Item>

										<DropdownMenu.Separator />

										<DropdownMenu.Item
											class="flex cursor-pointer items-center gap-2 text-destructive"
											onclick={handleDelete}
										>
											<Trash2 class="h-4 w-4" />
											<span>Delete</span>
										</DropdownMenu.Item>
									</DropdownMenu.Content>
								</DropdownMenu.Root>
							</div>

							<!-- Content section -->
							<div
								class="relative transition-all duration-300 {isExpanded
									? 'cursor-text select-text'
									: 'select-none'}"
								style={!isExpanded && showExpandButton
									? `max-height: ${MAX_HEIGHT}px;`
									: 'max-height: none;'}
							>
								{#if currentConfig.renderUserContentAsMarkdown}
									<div bind:this={contentElement}>
										<MarkdownContent
											class="markdown-system-content -my-4"
											content={content}
										/>
									</div>
								{:else}
									<span
										bind:this={contentElement}
										class="text-md whitespace-pre-wrap {isExpanded ? 'cursor-text' : ''}"
									>
										{content}
									</span>
								{/if}

								{#if !isExpanded && showExpandButton}
									<div
										class="pointer-events-none absolute right-0 bottom-0 left-0 h-48 bg-gradient-to-t from-muted to-transparent"
									></div>

									<div
										class="pointer-events-none absolute right-0 bottom-4 left-0 flex justify-center opacity-0 transition-opacity group-hover/expand:opacity-100"
									>
										<Button
											class="rounded-full px-4 py-1.5 text-xs shadow-md"
											size="sm"
											variant="outline"
										>
											Show full instruction
										</Button>
									</div>
								{/if}
							</div>

							{#if isExpanded && showExpandButton}
								<div class="mb-2 flex justify-center">
									<Button
										class="rounded-full px-4 py-1.5 text-xs"
										onclick={(e) => {
											e.stopPropagation();
											toggleExpand();
										}}
										size="sm"
										variant="outline"
									>
										Collapse Instruction
									</Button>
								</div>
							{/if}

							<div class="mt-3 flex justify-end">
								<Button size="sm" class="gap-1.5" onclick={handleStartNewChat}>
									<ArrowRight class="h-3.5 w-3.5" />
									Start new chat
								</Button>
							</div>
						</div>
					</div>
				</button>
			</div>
		{:else}
			<!-- Empty content: still show title/date and actions inside the dashed box -->
			<div class="relative">
				<div
					class="relative flex flex-col justify-between rounded-[1.125rem] border-2 border-dashed border-border/50 bg-muted px-6 py-4 min-h-[6rem]"
				>
					<div class="flex items-start justify-between gap-2">
						<div class="min-w-0">
							<h3 class="text-xl font-semibold tracking-tight text-foreground">{title}</h3>
							<p class="mt-0.5 text-xs text-muted-foreground">{formattedDate}</p>
						</div>
						<DropdownMenu.Root>
							<DropdownMenu.Trigger>
								<Tooltip.Root>
									<Tooltip.Trigger>
										<Button
											variant="ghost"
											size="sm"
											class="h-6 w-6 p-0 hover:bg-transparent data-[state=open]:bg-transparent!"
											aria-label="More actions"
										>
											<MoreHorizontal class="h-3 w-3" />
										</Button>
									</Tooltip.Trigger>
									<Tooltip.Content side="bottom">
										<p>More actions</p>
									</Tooltip.Content>
								</Tooltip.Root>
							</DropdownMenu.Trigger>
							<DropdownMenu.Content class="w-32">
								<DropdownMenu.Item
									class="flex cursor-pointer items-center gap-2"
									onclick={() => onEdit?.(id)}
								>
									<Edit class="h-4 w-4" />
									<span>Edit</span>
								</DropdownMenu.Item>

								<DropdownMenu.Separator />

								<DropdownMenu.Item
									class="flex cursor-pointer items-center gap-2 text-destructive"
									onclick={handleDelete}
								>
									<Trash2 class="h-4 w-4" />
									<span>Delete</span>
								</DropdownMenu.Item>
							</DropdownMenu.Content>
						</DropdownMenu.Root>
					</div>
					<div class="mt-3 flex justify-end">
						<Button size="sm" class="gap-1.5" onclick={handleStartNewChat}>
							<ArrowRight class="h-3.5 w-3.5" />
							Start new chat
						</Button>
					</div>
				</div>
			</div>
		{/if}
	{:else}
		<div class="space-y-3 rounded-[1.125rem] border-2 border-dashed border-border/50 bg-muted px-6 py-4">
			<div class="space-y-1">
				<label class="text-xs font-medium text-muted-foreground">Title</label>
				<Input
					class="text-foreground"
					type="text"
					bind:value={editTitle}
					placeholder="Instruction title"
				/>
				{#if editTitleError}
					<p class="text-xs text-destructive">{editTitleError}</p>
				{/if}
			</div>

			<div class="space-y-1">
				<label class="text-xs font-medium text-muted-foreground">Content</label>
				<textarea
					class="flex min-h-[100px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
					placeholder="Instruction content..."
					bind:value={editContent}
				/>
			</div>

			<div class="flex justify-end gap-2">
				<Button size="sm" variant="outline" onclick={handleCancel}>
					Cancel
				</Button>
				<Button
					size="sm"
					onclick={handleSave}
					disabled={!!editTitleError}
				>
					<Check class="mr-1 h-3 w-3" />
					Save
				</Button>
			</div>
		</div>
	{/if}
</div>
