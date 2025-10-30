<script lang="ts">
	import { X } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { formatFileSize, getFileTypeLabel, getPreviewText } from '$lib/utils/file-preview';
	import { FileTypeCategory, MimeTypeText } from '$lib/enums/files';

	interface Props {
		class?: string;
		id: string;
		onClick?: (event?: MouseEvent) => void;
		onRemove?: (id: string) => void;
		name: string;
		readonly?: boolean;
		size?: number;
		textContent?: string;
		type: string;
	}

	let {
		class: className = '',
		id,
		onClick,
		onRemove,
		name,
		readonly = false,
		size,
		textContent,
		type
	}: Props = $props();
</script>

{#if type === MimeTypeText.PLAIN || type === FileTypeCategory.TEXT}
	{#if readonly}
		<!-- Readonly mode (ChatMessage) -->
		<button
			class="cursor-pointer rounded-lg border border-border bg-muted p-3 transition-shadow hover:shadow-md {className} w-full max-w-2xl"
			onclick={onClick}
			aria-label={`Preview ${name}`}
			type="button"
		>
			<div class="flex items-start gap-3">
				<div class="flex min-w-0 flex-1 flex-col items-start text-left">
					<span class="w-full truncate text-sm font-medium text-foreground">{name}</span>

					{#if size}
						<span class="text-xs text-muted-foreground">{formatFileSize(size)}</span>
					{/if}

					{#if textContent && type === 'text'}
						<div class="relative mt-2 w-full">
							<div
								class="overflow-hidden font-mono text-xs leading-relaxed break-words whitespace-pre-wrap text-muted-foreground"
							>
								{getPreviewText(textContent)}
							</div>

							{#if textContent.length > 150}
								<div
									class="pointer-events-none absolute right-0 bottom-0 left-0 h-6 bg-gradient-to-t from-muted to-transparent"
								></div>
							{/if}
						</div>
					{/if}
				</div>
			</div>
		</button>
	{:else}
		<!-- Non-readonly mode (ChatForm) -->
		<div class="relative rounded-lg border border-border bg-muted p-3 {className} w-64">
			<Button
				type="button"
				variant="ghost"
				size="sm"
				class="absolute top-2 right-2 h-6 w-6 bg-white/20 p-0 hover:bg-white/30"
				onclick={() => onRemove?.(id)}
				aria-label="Remove file"
			>
				<X class="h-3 w-3" />
			</Button>

			<div class="pr-8">
				<span class="mb-3 block truncate text-sm font-medium text-foreground">{name}</span>

				{#if textContent}
					<div class="relative">
						<div
							class="overflow-hidden font-mono text-xs leading-relaxed break-words whitespace-pre-wrap text-muted-foreground"
							style="max-height: 3.6em; line-height: 1.2em;"
						>
							{getPreviewText(textContent)}
						</div>

						{#if textContent.length > 150}
							<div
								class="pointer-events-none absolute right-0 bottom-0 left-0 h-4 bg-gradient-to-t from-muted to-transparent"
							></div>
						{/if}
					</div>
				{/if}
			</div>
		</div>
	{/if}
{:else}
	<button
		class="flex items-center gap-2 gap-3 rounded-lg border border-border bg-muted p-3 {className}"
		onclick={onClick}
	>
		<div
			class="flex h-8 w-8 items-center justify-center rounded bg-primary/10 text-xs font-medium text-primary"
		>
			{getFileTypeLabel(type)}
		</div>

		<div class="flex flex-col gap-1">
			<span class="max-w-36 truncate text-sm font-medium text-foreground md:max-w-72">
				{name}
			</span>

			{#if size}
				<span class="text-left text-xs text-muted-foreground">{formatFileSize(size)}</span>
			{/if}
		</div>

		{#if !readonly}
			<Button
				type="button"
				variant="ghost"
				size="sm"
				class="h-6 w-6 p-0"
				onclick={(e) => {
					e.stopPropagation();
					onRemove?.(id);
				}}
			>
				<X class="h-3 w-3" />
			</Button>
		{/if}
	</button>
{/if}
