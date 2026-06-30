<script lang="ts">
	import { MarkdownContent, DropdownMenuActions } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import Input from '$lib/components/ui/input/input.svelte';
	import Label from '$lib/components/ui/label/label.svelte';
	import { Textarea } from '$lib/components/ui/textarea';
	import { config } from '$lib/stores/settings.svelte';
	import { skillsStore } from '$lib/stores/skills.svelte';
	import { skillPreferencesStore } from '$lib/stores/skill-preferences.svelte';
	import { untrack } from 'svelte';
	import { MoreHorizontal, Trash2, Edit, ArrowRight, Download } from '@lucide/svelte';
	import {
		validateSkillName,
		validateSkillDescription,
		normalizeSkillName
	} from '$lib/utils/skill-format';
	import type { SkillOrigin } from '$lib/types';

	interface Props {
		id: string;
		name: string;
		description: string;
		content: string;
		alwaysOn?: boolean;
		lastModified: number;
		path?: string;
		origin?: SkillOrigin;
		onDelete?: () => void;
		onEdit?: (id: string) => void;
		onStartNewChat?: (id: string, name: string, content: string) => void;
	}

	let {
		id,
		name,
		description,
		content,
		alwaysOn = false,
		lastModified,
		path,
		origin = 'lib',
		onDelete,
		onEdit,
		onStartNewChat
	}: Props = $props();

	// The `alwaysOn` prop is only used as a default when the prefs store
	// hasn't hydrated yet (e.g. SSR). After hydration the prefs store is
	// the source of truth and the prop is ignored.
	let isAlwaysOn = $derived.by(() => {
		const hydrated = skillPreferencesStore.getAlwaysOnIds();
		// If the prefs store has any always-on set the row must be one of them
		// to count as always-on. Until hydration finishes we fall back to the
		// prop so the initial render isn't a flash of "off".
		if (hydrated.length === 0 && alwaysOn) return true;
		return skillPreferencesStore.isAlwaysOn(id);
	});

	// Edit state (initialized from props; refilled when props change below)
	let isEditing = $state(false);
	let editName = $state(untrack(() => name));
	let editDescription = $state(untrack(() => description));
	let editContent = $state(untrack(() => content));

	let existingNames = $derived(
		skillsStore
			.getSkills()
			.filter((s) => s.id !== id)
			.map((s) => ({ id: s.id, name: s.name }))
	);
	let editNameError = $derived.by(() => {
		const normalized = normalizeSkillName(editName);
		const nameErr = validateSkillName(normalized);
		if (nameErr) return nameErr;
		const dup = existingNames.find(
			(s) => normalizeSkillName(s.name).toLowerCase() === normalized.toLowerCase()
		);
		return dup ? 'A skill with this name already exists' : null;
	});
	// Description is always required (Agent Skills spec, llama-ui always-on
	// gate, and Pi's auto-load filter). Empty descriptions are rejected here
	// as well as at the store boundary so users see the failure inline.
	let editDescriptionError = $derived.by(() => validateSkillDescription(editDescription));
	let editContentError = $derived.by(() => (!editContent.trim() ? 'Content is required' : null));
	let editSaveError = $derived.by(() => editNameError || editContentError || editDescriptionError);

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

	// Sync edit values when props change externally
	$effect(() => {
		if (!isEditing) {
			editName = name;
			editDescription = description;
			editContent = content;
		}
	});

	async function toggleAlwaysOn() {
		if (!skillPreferencesStore.isAlwaysOn(id) && !description.trim()) return;
		skillPreferencesStore.toggleAlwaysOn(id);
	}

	function toggleExpand() {
		isExpanded = !isExpanded;
	}

	async function handleSave() {
		if (editSaveError) return;
		const updates: Parameters<typeof skillsStore.updateSkill>[1] = {
			name: normalizeSkillName(editName),
			description: editDescription.trim(),
			content: editContent.trim()
		};
		await skillsStore.updateSkill(id, updates);
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
		onStartNewChat?.(id, name, content);
	}

	let formattedDate = $derived.by(() => {
		const date = new Date(lastModified);
		return date.toLocaleDateString('en-US', {
			year: 'numeric',
			month: 'short',
			day: 'numeric'
		});
	});

	function handleExport() {
		const markdown = skillsStore.exportSkillMarkdown(id);
		if (!markdown) return;

		const blob = new Blob([markdown], { type: 'text/markdown' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `llama_skill_${name.replace(/\s+/g, '-').toLowerCase()}_${new Date().toISOString().split('T')[0]}.md`;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
	}

	const moreActions = [
		{ icon: Edit, label: 'Edit', onclick: startEdit },
		{ icon: Download, label: 'Export', onclick: handleExport },
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
							<!-- Top bar: name, description, date, more menu -->
							<div class="mb-2 flex items-start justify-between gap-2">
								<div class="min-w-0">
									<div class="flex items-center gap-2">
										<h3 class="text-xl font-semibold tracking-tight text-foreground">{name}</h3>
										{#if isAlwaysOn}
											<span
												class="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-primary"
											>
												always-on
											</span>
										{/if}
										{#if origin !== 'lib'}
											<span
												class="rounded-full bg-muted px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground"
												title={path ?? ''}
											>
												{origin}
											</span>
										{/if}
									</div>
									<p class="mt-0.5 text-xs text-muted-foreground">{formattedDate}</p>
									<p class="mt-1 text-sm text-muted-foreground">{description}</p>
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
										Collapse Skill
									</Button>
								</div>
							{/if}

							<div class="mt-3 flex items-center justify-between gap-2">
								<label class="inline-flex items-center gap-2 text-xs text-muted-foreground">
									<input
										type="checkbox"
										checked={isAlwaysOn}
										disabled={!description.trim()}
										onclick={(e) => e.stopPropagation()}
										onchange={toggleAlwaysOn}
										class="h-3.5 w-3.5 rounded border-input"
									/>
									<span>Always include in system prompt</span>
								</label>
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
			<!-- Empty content: still show name/date and actions inside the dashed box -->
			<div class="relative">
				<div
					class="relative flex flex-col justify-between rounded-[1.125rem] border-2 border-dashed border-border/50 bg-muted p-4 min-h-24"
				>
					<div class="flex items-start justify-between gap-2">
						<div class="min-w-0">
							<div class="flex items-center gap-2">
								<h3 class="text-xl font-semibold tracking-tight text-foreground">{name}</h3>
								{#if isAlwaysOn}
									<span
										class="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-primary"
									>
										always-on
									</span>
								{/if}
							</div>
							<p class="mt-0.5 text-xs text-muted-foreground">{formattedDate}</p>
							<p class="mt-1 text-sm text-muted-foreground">{description}</p>
						</div>
						<DropdownMenuActions
							triggerIcon={MoreHorizontal}
							triggerTooltip="More actions"
							actions={moreActions}
						/>
					</div>
				</div>
			</div>
		{/if}
	{:else}
		<div class="space-y-6 rounded-[1.125rem] border-2 border-dashed border-border/50 bg-muted p-4">
			<div class="space-y-2">
				<Label for="skill-edit-name" class="text-sm font-medium">Name</Label>

				<Input
					id="skill-edit-name"
					type="text"
					bind:value={editName}
					placeholder="e.g. Code Review Assistant"
					class="w-full"
				/>

				{#if editNameError}
					<p class="text-xs text-destructive">{editNameError}</p>
				{/if}
			</div>

			<div class="space-y-2">
				<Label for="skill-edit-description" class="text-sm font-medium">
					Description <span class="text-destructive">*</span>
				</Label>

				<Input
					id="skill-edit-description"
					type="text"
					bind:value={editDescription}
					placeholder="A short summary suitable for the system prompt"
					class="w-full"
				/>

				{#if editDescriptionError}
					<p class="text-xs text-destructive">{editDescriptionError}</p>
				{/if}
			</div>

			<div class="space-y-2">
				<Label for="skill-edit-content" class="text-sm font-medium">Content</Label>

				<Textarea
					id="skill-edit-content"
					bind:value={editContent}
					placeholder="Enter skill content..."
					class="min-h-[10rem] w-full"
				/>

				{#if editContentError}
					<p class="text-xs text-destructive">{editContentError}</p>
				{/if}
			</div>

			<div class="flex items-center justify-end gap-2">
				<Button variant="secondary" size="sm" onclick={handleCancel}>Cancel</Button>

				<Button size="sm" onclick={handleSave} disabled={!!editSaveError}>Save</Button>
			</div>
		</div>
	{/if}
</div>
