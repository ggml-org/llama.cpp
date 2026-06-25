<script lang="ts">
	import { MarkdownContent, CategoryCombobox, DropdownMenuActions } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import Input from '$lib/components/ui/input/input.svelte';
	import Label from '$lib/components/ui/label/label.svelte';
	import { Textarea } from '$lib/components/ui/textarea';
	import Badge from '$lib/components/ui/badge/badge.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { promptsStore } from '$lib/stores/prompts.svelte';
	import { MoreHorizontal, Trash2, Edit, ArrowRight } from '@lucide/svelte';

	interface Props {
		id: string;
		title: string;
		content: string;
		lastModified: number;
		category?: string;
		onDelete?: () => void;
		onEdit?: (id: string) => void;
		onStartNewChat?: (id: string, title: string, content: string) => void;
	}

	let { id, title, content, lastModified, category, onDelete, onEdit, onStartNewChat }: Props =
		$props();

	// Edit state
	let isEditing = $state(false);
	let editTitle = $state(title);
	let editContent = $state(content);
	let editCategory = $state(category ?? '');
	let existingCategories = $derived(promptsStore.getCategories());
	let editTitleError = $derived.by(() => (!editTitle.trim() ? 'Title is required' : null));
	let editContentError = $derived.by(() => (!editContent.trim() ? 'Content is required' : null));
	let editSaveError = $derived.by(() => editTitleError || editContentError);
	let displayCategory = $derived(category && category.trim() ? category.trim() : null);

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
			editCategory = category ?? '';
		}
	});

	function toggleExpand() {
		isExpanded = !isExpanded;
	}

	async function handleSave() {
		if (editSaveError) return;
		const trimmedCategory = editCategory.trim();
		await promptsStore.updatePrompt(id, {
			title: editTitle.trim(),
			content: editContent.trim(),
			...(trimmedCategory ? { category: trimmedCategory } : { category: undefined })
		});
		isEditing = false;
		onEdit?.(id);
	}

	function startEdit() {
		isEditing = true;
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

	const moreActions = [
		{ icon: Edit, label: 'Edit', onclick: startEdit },
		{
			icon: Trash2,
			label: 'Delete',
			onclick: handleDelete,
			variant: 'destructive' as const,
			separator: true
		}
	];
</script>

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
							class="relative flex flex-col justify-between rounded-[1.125rem] border-2 border-dashed border-border/50 bg-muted p-4"
							style="overflow-wrap: anywhere; word-break: break-word; min-height: 6rem;"
						>
							<!-- Top bar: title, date, more menu -->
							<div class="mb-2 flex items-start justify-between gap-2">
								<div class="min-w-0">
									<h3 class="text-xl font-semibold tracking-tight text-foreground">{title}</h3>
									<div class="mt-0.5 flex items-center gap-2">
										<p class="text-xs text-muted-foreground">{formattedDate}</p>
										{#if displayCategory}
											<Badge variant="secondary" class="text-[0.65rem] font-medium"
												>{displayCategory}</Badge
											>
										{/if}
									</div>
								</div>
								<DropdownMenuActions
									triggerIcon={MoreHorizontal}
									triggerTooltip="More actions"
									actions={moreActions}
								/>
							</div>

							<!-- Content section -->
							<div
								class="relative overflow-hidden transition-all duration-300 {isExpanded
									? 'cursor-text select-text'
									: 'select-none'}"
								style={!isExpanded && showExpandButton
									? `max-height: ${MAX_HEIGHT}px;`
									: 'max-height: none;'}
							>
								{#if currentConfig.renderUserContentAsMarkdown}
									<div bind:this={contentElement}>
										<MarkdownContent class="markdown-system-content -my-4" {content} />
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
										Collapse Prompt
									</Button>
								</div>
							{/if}

							<div class="mt-3 flex justify-end">
								<Button size="sm" class="gap-1.5" onclick={handleStartNewChat}>
									New chat
									<ArrowRight class="h-3.5 w-3.5" />
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
					class="relative flex flex-col justify-between rounded-[1.125rem] border-2 border-dashed border-border/50 bg-muted p-4 min-h-24"
				>
					<div class="flex items-start justify-between gap-2">
						<div class="min-w-0">
							<h3 class="text-xl font-semibold tracking-tight text-foreground">{title}</h3>
							<div class="mt-0.5 flex items-center gap-2">
								<p class="text-xs text-muted-foreground">{formattedDate}</p>
								{#if displayCategory}
									<Badge variant="secondary" class="text-[0.65rem] font-medium"
										>{displayCategory}</Badge
									>
								{/if}
							</div>
						</div>
						<DropdownMenuActions
							triggerIcon={MoreHorizontal}
							triggerTooltip="More actions"
							actions={moreActions}
						/>
					</div>
					<div class="mt-3 flex justify-end">
						<Button size="sm" class="gap-1.5" onclick={handleStartNewChat}>
							New chat
							<ArrowRight class="h-3.5 w-3.5" />
						</Button>
					</div>
				</div>
			</div>
		{/if}
	{:else}
		<div class="space-y-6 rounded-[1.125rem] border-2 border-dashed border-border/50 bg-muted p-4">
			<div class="space-y-2">
				<Label for="prompt-edit-title" class="text-sm font-medium">Title</Label>

				<Input
					id="prompt-edit-title"
					type="text"
					bind:value={editTitle}
					placeholder="e.g. Code Review Assistant"
					class="w-full"
				/>

				{#if editTitleError}
					<p class="text-xs text-destructive">{editTitleError}</p>
				{/if}
			</div>

			<div class="space-y-2">
				<Label for="prompt-edit-content" class="text-sm font-medium">Content</Label>

				<Textarea
					id="prompt-edit-content"
					bind:value={editContent}
					placeholder="Enter prompt content..."
					class="min-h-[10rem] w-full"
				/>

				{#if editContentError}
					<p class="text-xs text-destructive">{editContentError}</p>
				{/if}
			</div>

			<div class="flex flex-col gap-2">
				<label for="prompt-edit-category" class="text-sm font-medium">Category</label>

				<CategoryCombobox
					id="prompt-edit-category"
					bind:value={editCategory}
					categories={existingCategories}
				/>
			</div>

			<div class="flex items-center justify-end gap-2">
				<Button variant="secondary" size="sm" onclick={handleCancel}>Cancel</Button>

				<Button size="sm" onclick={handleSave} disabled={!!editSaveError}>Save</Button>
			</div>
		</div>
	{/if}
</div>
