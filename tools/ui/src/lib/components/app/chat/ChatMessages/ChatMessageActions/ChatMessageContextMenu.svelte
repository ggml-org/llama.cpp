<script lang="ts">
	import { ArrowRight, Copy, Edit, GitBranch, RefreshCw, Trash2 } from '@lucide/svelte';
	import { ContextMenuActions } from '$lib/components/app';
	import { NATIVE_CONTEXT_MENU_SELECTOR } from '$lib/constants';
	import { getChatMessageActionsContext, getChatMessageEditContext } from '$lib/contexts';
	import { MessageRole } from '$lib/enums';
	import type { MenuAction } from '$lib/types';
	import type { Snippet } from 'svelte';

	interface Props {
		role: MessageRole.USER | MessageRole.ASSISTANT;
		children: Snippet;
		disabled?: boolean;
		onRegenerate?: () => void;
		onContinue?: () => void;
	}

	let { children, disabled = false, onContinue, onRegenerate, role }: Props = $props();

	const messageActions = getChatMessageActionsContext();
	const editCtx = getChatMessageEditContext();

	// the same actions as the message's action icons
	let actions = $derived<MenuAction[]>([
		{ icon: Copy, label: 'Copy', onclick: messageActions.copy },
		{ icon: Edit, label: 'Edit', onclick: editCtx.startEdit },
		...(role === MessageRole.ASSISTANT && onRegenerate
			? [{ icon: RefreshCw, label: 'Regenerate', onclick: () => onRegenerate() }]
			: []),
		...(role === MessageRole.ASSISTANT && onContinue
			? [{ icon: ArrowRight, label: 'Continue', onclick: () => onContinue() }]
			: []),
		...(messageActions.forkConversation
			? [
					{
						icon: GitBranch,
						label: 'Fork conversation',
						onclick: () => messageActions.setShowForkDialog(true)
					}
				]
			: []),
		{
			icon: Trash2,
			label: 'Delete',
			onclick: messageActions.requestDelete,
			separator: true,
			variant: 'destructive'
		}
	]);

	// Selected text, links, media, code and fields keep the browser's menu: the
	// right-click stops here, before the context menu trigger around it sees it.
	function keepNativeMenu(event: MouseEvent) {
		const hasSelection = Boolean(window.getSelection()?.toString().trim());
		const target = event.target instanceof Element ? event.target : null;

		if (hasSelection || target?.closest(NATIVE_CONTEXT_MENU_SELECTOR)) {
			event.stopPropagation();
		}
	}
</script>

<ContextMenuActions {actions} {disabled}>
	<div class="contents" oncontextmenucapture={keepNativeMenu}>
		{@render children()}
	</div>
</ContextMenuActions>
