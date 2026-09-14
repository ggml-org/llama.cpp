<script lang="ts">
	import { ChevronDown, ChevronRight } from '@lucide/svelte';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import { Input } from '$lib/components/ui/input';
	import * as Select from '$lib/components/ui/select';
	import { DEFAULT_BACKEND_CHAT_PATH, DEFAULT_BACKEND_MODELS_PATH } from '$lib/constants';
	import type { Backend, BackendProtocol } from '$lib/types';

	const PROTOCOL_OPTIONS: Array<{ label: string; value: BackendProtocol }> = [
		{ label: 'OpenAI-compatible', value: 'openai' },
		{ label: 'Anthropic-compatible', value: 'anthropic' },
		{ label: 'llama.cpp (llama-server)', value: 'llama.cpp' }
	];

	interface Props {
		backend: Backend;
		id: string;
		onChange: (patch: Partial<Backend>) => void;
		urlError?: string | null;
	}

	let { backend, id, onChange, urlError = null }: Props = $props();

	let showAdvanced = $state(false);

	let protocolLabel = $derived(
		PROTOCOL_OPTIONS.find((option) => option.value === backend.protocol)?.label ?? ''
	);
	let isAnthropic = $derived(backend.protocol === 'anthropic');
</script>

<div class="grid gap-2">
	<div>
		<label class="mb-2 block text-xs font-medium select-none" for="backend-url-{id}">
			Base URL <span class="text-destructive">*</span>
		</label>

		<Input
			class={urlError ? 'border-destructive' : ''}
			id="backend-url-{id}"
			oninput={(e) => onChange({ baseUrl: e.currentTarget.value })}
			placeholder="https://api.example.com"
			type="url"
			value={backend.baseUrl}
		/>

		{#if urlError}
			<p class="mt-1.5 text-xs text-destructive">{urlError}</p>
		{/if}
	</div>

	<div>
		<label class="mb-2 block text-xs font-medium select-none" for="backend-name-{id}">
			Display name
		</label>

		<Input
			id="backend-name-{id}"
			oninput={(e) => onChange({ name: e.currentTarget.value })}
			placeholder="Name shown in the model selector"
			type="text"
			value={backend.name}
		/>
	</div>

	<div>
		<span class="mb-2 block text-xs font-medium select-none">API format</span>

		<Select.Root
			onValueChange={(value) => onChange({ protocol: value as BackendProtocol })}
			type="single"
			value={backend.protocol}
		>
			<Select.Trigger class="w-full">{protocolLabel}</Select.Trigger>

			<Select.Content>
				{#each PROTOCOL_OPTIONS as option (option.value)}
					<Select.Item label={option.label} value={option.value}>{option.label}</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
	</div>

	<div>
		<label class="mb-2 block text-xs font-medium select-none" for="backend-key-{id}">
			API key
		</label>

		<Input
			autocomplete="off"
			id="backend-key-{id}"
			oninput={(e) => onChange({ apiKey: e.currentTarget.value || undefined })}
			placeholder="Optional"
			type="password"
			value={backend.apiKey ?? ''}
		/>

		<p class="mt-1.5 text-xs text-muted-foreground">
			{#if isAnthropic}
				Sent as the x-api-key header.
			{:else}
				Sent as a Bearer token.
			{/if}
		</p>
	</div>

	<Collapsible.Root bind:open={showAdvanced}>
		<Collapsible.Trigger
			class="flex items-center gap-1 text-xs font-medium text-muted-foreground select-none hover:text-foreground"
		>
			{#if showAdvanced}
				<ChevronDown class="h-3.5 w-3.5" />
			{:else}
				<ChevronRight class="h-3.5 w-3.5" />
			{/if}

			Advanced
		</Collapsible.Trigger>

		<Collapsible.Content>
			<div class="mt-3 grid gap-4">
				<div>
					<label class="mb-2 block text-xs font-medium select-none" for="backend-chat-path-{id}">
						Chat completions path
					</label>

					<Input
						id="backend-chat-path-{id}"
						oninput={(e) => onChange({ chatPath: e.currentTarget.value || undefined })}
						placeholder={DEFAULT_BACKEND_CHAT_PATH}
						type="text"
						value={backend.chatPath ?? ''}
					/>
				</div>

				<div>
					<label class="mb-2 block text-xs font-medium select-none" for="backend-models-path-{id}">
						Models path
					</label>

					<Input
						id="backend-models-path-{id}"
						oninput={(e) => onChange({ modelsPath: e.currentTarget.value || undefined })}
						placeholder={DEFAULT_BACKEND_MODELS_PATH}
						type="text"
						value={backend.modelsPath ?? ''}
					/>
				</div>
			</div>
		</Collapsible.Content>
	</Collapsible.Root>
</div>
