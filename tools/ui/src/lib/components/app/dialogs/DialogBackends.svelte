<script lang="ts">
	import { SettingsBackends } from '$lib/components/app/backends';
	import { Logo } from '$lib/components/app/misc';
	import * as Dialog from '$lib/components/ui/dialog';
	import { cn } from '$lib/components/ui/utils';
	import type { BackendProtocol } from '$lib/types';

	const PROTOCOL_TABS: Array<{ label: string; value: BackendProtocol }> = [
		{ label: 'Llama-compat', value: 'llama.cpp' },
		{ label: 'OpenAI-compat', value: 'openai' }
	];

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { onOpenChange, open = $bindable(false) }: Props = $props();

	let protocol = $state<BackendProtocol>('llama.cpp');

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}
</script>

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content class="md:max-h-[80vh]! md:w-[calc(100vw-4rem)]! md:max-w-164! flex flex-col">
		<Dialog.Header>
			<Dialog.Title class="flex items-center gap-2">
				<Logo class="h-5 w-5" style="--size: 1.25rem" />

				<span>Backends</span>
			</Dialog.Title>

			<Dialog.Description>
				Endpoints this UI can talk to. The built-in backend is the llama.cpp server serving it.
			</Dialog.Description>
		</Dialog.Header>

		<div class="mt-4 flex w-fit shrink-0 items-center gap-1 rounded-full bg-muted p-1">
			{#each PROTOCOL_TABS as tab (tab.value)}
				<button
					aria-pressed={protocol === tab.value}
					class={cn(
						'cursor-pointer rounded-full px-3 py-1 text-xs font-medium transition-colors',
						protocol === tab.value
							? 'bg-background text-foreground shadow-sm'
							: 'text-muted-foreground hover:text-foreground'
					)}
					onclick={() => (protocol = tab.value)}
					type="button"
				>
					{tab.label}
				</button>
			{/each}
		</div>

		<SettingsBackends class="mt-2 overflow-y-auto" {protocol} />
	</Dialog.Content>
</Dialog.Root>
