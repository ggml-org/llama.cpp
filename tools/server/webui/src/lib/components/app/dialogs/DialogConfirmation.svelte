<script lang="ts">
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import type { Component } from 'svelte';
	import { KeyboardKey } from '$lib/enums';

	interface Props {
		open: boolean;
		title: string;
		description: string;
		confirmText?: string;
		cancelText?: string;
		variant?: 'default' | 'destructive';
		icon?: Component;
		onConfirm: () => void;
		onCancel: () => void;
		onKeydown?: (event: KeyboardEvent) => void;
	}

	let {
		open = $bindable(),
		title,
		description,
		confirmText = 'Confirm',
		cancelText = 'Cancel',
		variant = 'default',
		icon,
		onConfirm,
		onCancel,
		onKeydown
	}: Props = $props();

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === KeyboardKey.ENTER) {
			event.preventDefault();
			onConfirm();
		}
		onKeydown?.(event);
	}

	function handleOpenChange(newOpen: boolean) {
		if (!newOpen) {
			onCancel();
		}
	}
</script>

<AlertDialog.Root {open} onOpenChange={handleOpenChange}>
	<AlertDialog.Content onkeydown={handleKeydown}>
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				{#if icon}
					{@const IconComponent = icon}
					<IconComponent class="h-5 w-5 {variant === 'destructive' ? 'text-destructive' : ''}" />
				{/if}
				{title}
			</AlertDialog.Title>

			<AlertDialog.Description>
				{description}
			</AlertDialog.Description>
		</AlertDialog.Header>

		<AlertDialog.Footer>
			<AlertDialog.Cancel onclick={onCancel}>{cancelText}</AlertDialog.Cancel>
			<AlertDialog.Action
				onclick={onConfirm}
				class={variant === 'destructive' ? 'bg-destructive text-white hover:bg-destructive/80' : ''}
			>
				{confirmText}
			</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
