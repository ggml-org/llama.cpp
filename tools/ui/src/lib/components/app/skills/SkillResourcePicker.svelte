<script lang="ts">
	import { ChevronRight, FileCode2, FileText, Folder, FolderOpen } from '@lucide/svelte';
	import * as Popover from '$lib/components/ui/popover';
	import {
		buildSkillResourceTree,
		createSkillRootNode,
		flattenSkillResourceTree,
		getInitialExpandedFolderPaths,
		type SkillResourceTreeNode
	} from './skill-resource-presentation';

	interface Props {
		paths: readonly string[];
		resourceCount: number;
		resourcesTruncated: boolean;
		selectedPath: string;
		unavailablePaths: ReadonlySet<string>;
		onSelect: (path: string) => void;
	}

	let {
		paths,
		resourceCount,
		resourcesTruncated,
		selectedPath,
		unavailablePaths,
		onSelect
	}: Props = $props();
	let open = $state(false);
	let expandedPaths = $state<ReadonlySet<string>>(new Set());
	let rowButtons = $state<HTMLButtonElement[]>([]);

	const tree = $derived([createSkillRootNode(), ...buildSkillResourceTree(paths)]);
	const rows = $derived(flattenSkillResourceTree(tree, expandedPaths));
	const selectedNode = $derived(
		rows.find((row) => row.node.path === selectedPath)?.node ?? tree[0]
	);
	const hasResources = $derived(paths.length > 0);

	$effect(() => {
		void tree;
		expandedPaths = getInitialExpandedFolderPaths(tree);
	});

	function closeAndRestoreFocus() {
		open = false;
		requestAnimationFrame(() => document.getElementById('skill-resource-picker-trigger')?.focus());
	}

	function toggleFolder(path: string) {
		const next = new Set(expandedPaths);

		if (next.has(path)) next.delete(path);
		else next.add(path);

		expandedPaths = next;
	}

	function selectFile(node: SkillResourceTreeNode) {
		if (node.kind !== 'file' || node.format === 'unsupported' || unavailablePaths.has(node.path))
			return;

		onSelect(node.path);
		closeAndRestoreFocus();
	}

	function moveFocus(index: number) {
		requestAnimationFrame(() => rowButtons[index]?.focus());
	}

	function handleRowKeydown(event: KeyboardEvent, index: number, node: SkillResourceTreeNode) {
		if (event.key === 'ArrowDown') {
			event.preventDefault();
			moveFocus(Math.min(index + 1, rows.length - 1));
			return;
		}

		if (event.key === 'ArrowUp') {
			event.preventDefault();
			moveFocus(Math.max(index - 1, 0));
			return;
		}

		if (event.key === 'Home' || event.key === 'End') {
			event.preventDefault();
			moveFocus(event.key === 'Home' ? 0 : rows.length - 1);
			return;
		}

		if (event.key === 'ArrowRight' && node.kind === 'folder') {
			event.preventDefault();
			if (!expandedPaths.has(node.path)) toggleFolder(node.path);
			else moveFocus(index + 1);
			return;
		}

		if (event.key === 'ArrowLeft' && node.kind === 'folder' && expandedPaths.has(node.path)) {
			event.preventDefault();
			toggleFolder(node.path);
			return;
		}

		if (event.key === 'ArrowLeft') {
			const parentIndex = rows.findIndex((row) => row.node.path === rows[index].parentPath);
			if (parentIndex >= 0) {
				event.preventDefault();
				moveFocus(parentIndex);
			}
			return;
		}

		if (event.key === 'Escape') {
			event.preventDefault();
			closeAndRestoreFocus();
		}
	}
</script>

<Popover.Root {open} onOpenChange={(nextOpen) => (open = nextOpen && hasResources)}>
	<Popover.Trigger
		id="skill-resource-picker-trigger"
		data-testid="skill-resource-picker-trigger"
		class="inline-flex h-8 items-center gap-1.5 rounded-md border border-input bg-background px-3 text-sm font-medium shadow-xs transition-colors hover:bg-accent hover:text-accent-foreground"
	>
		<FileText class="size-3.5" aria-hidden="true" />
		<span class="max-w-48 truncate font-mono text-xs">{selectedNode.name}</span>
		<span class="text-muted-foreground">({resourceCount}{resourcesTruncated ? '+' : ''})</span>
	</Popover.Trigger>

	<Popover.Content class="w-80 p-1" align="start">
		{#if hasResources}
			<div
				role="tree"
				aria-label="Skill resources"
				data-testid="skill-resource-picker-tree"
				class="max-h-80 overflow-y-auto py-1"
			>
				{#each rows as row, index (row.node.path)}
					{@const unavailable =
						row.node.kind === 'file' &&
						(row.node.format === 'unsupported' || unavailablePaths.has(row.node.path))}
					<button
						bind:this={rowButtons[index]}
						type="button"
						role="treeitem"
						aria-level={row.depth + 1}
						aria-expanded={row.node.kind === 'folder'
							? expandedPaths.has(row.node.path)
							: undefined}
						aria-disabled={unavailable}
						tabindex={index === 0 ? 0 : -1}
						aria-selected={row.node.path === selectedPath}
						title={unavailable ? 'Preview unavailable' : undefined}
						class="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-sm hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
						class:bg-accent={row.node.path === selectedPath}
						disabled={unavailable}
						onclick={() =>
							row.node.kind === 'folder' ? toggleFolder(row.node.path) : selectFile(row.node)}
						onkeydown={(event) => handleRowKeydown(event, index, row.node)}
					>
						<span style:width={`${row.depth * 1.25}rem`} aria-hidden="true"></span>
						{#if row.node.kind === 'folder'}
							<ChevronRight
								class={`size-3.5 shrink-0 transition-transform ${expandedPaths.has(row.node.path) ? 'rotate-90' : ''}`}
								aria-hidden="true"
							/>
							{#if expandedPaths.has(row.node.path)}
								<FolderOpen class="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
							{:else}
								<Folder class="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
							{/if}
						{:else if row.node.format === 'markdown'}
							<FileText class="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
						{:else}
							<FileCode2 class="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
						{/if}
						<span class="min-w-0 truncate font-mono text-xs">{row.node.name}</span>
					</button>
				{/each}
			</div>
		{:else}
			<p class="px-2 py-1 text-xs text-muted-foreground">No additional resources</p>
		{/if}
	</Popover.Content>
</Popover.Root>
