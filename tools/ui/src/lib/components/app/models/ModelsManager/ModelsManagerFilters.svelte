<script lang="ts">
	import type { ModalityKey } from './utils';
	import { Check, ChevronDown, Image, Lightbulb, Mic, Server, Video, Wrench } from '@lucide/svelte';
	import { Logo, ScrollCarousel } from '$lib/components/app';
	import { BackendIcon } from '$lib/components/app/backends';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Select from '$lib/components/ui/select';
	import { Toggle } from '$lib/components/ui/toggle';
	import * as ToggleGroup from '$lib/components/ui/toggle-group';
	import { FILTER_TRIGGER_CLASS, LOCAL_BACKEND_ID } from '$lib/constants';
	import { ModelCapability } from '$lib/enums';
	import type { Backend } from '$lib/types/backend';

	interface Props {
		backends: Backend[];
		/** Capabilities a model must have every one of. */
		capabilities?: ModelCapability[];
		/** Smallest context a model must support; 0 keeps every model. */
		contextLimit?: number;
		/** Keep only models that have a draft sidecar to speculate with. */
		draft?: boolean;
		/** Modalities a model must support at least one of. */
		modalities?: ModalityKey[];
		/** Backend ids to keep; empty keeps every provider. */
		providers?: string[];
		/** Repos each provider contributes to the current search, shown in the menu. */
		providerCounts?: Record<string, number>;
	}

	let {
		backends,
		capabilities = $bindable<ModelCapability[]>([]),
		contextLimit = $bindable(0),
		draft = $bindable(false),
		modalities = $bindable<ModalityKey[]>([]),
		providerCounts = {},
		providers = $bindable<string[]>([])
	}: Props = $props();

	const CONTEXT_STEPS: { label: string; value: number }[] = [
		{ label: 'Any context', value: 0 },
		{ label: '8K or more', value: 8192 },
		{ label: '32K or more', value: 32768 },
		{ label: '128K or more', value: 131_072 },
		{ label: '256K or more', value: 262_144 },
		{ label: '1M or more', value: 1_048_576 }
	];
	// one group for everything a model either has or does not: what it can do, and
	// what it can accept
	const TOGGLES: {
		group: 'capability' | 'modality';
		icon: typeof Wrench;
		label: string;
		value: string;
	}[] = [
		{ group: 'capability', icon: Wrench, label: 'Tool use', value: ModelCapability.TOOL_USE },
		{ group: 'capability', icon: Lightbulb, label: 'Reasoning', value: ModelCapability.REASONING },
		{ group: 'modality', icon: Image, label: 'Vision', value: 'vision' },
		{ group: 'modality', icon: Video, label: 'Video', value: 'video' },
		{ group: 'modality', icon: Mic, label: 'Audio', value: 'audio' }
	];

	// the group holds one flat list, so a change splits back into the two filters
	function setToggles(values: string[]): void {
		capabilities = values.filter(
			(value) => TOGGLES.find((entry) => entry.value === value)?.group === 'capability'
		) as ModelCapability[];
		modalities = values.filter(
			(value) => TOGGLES.find((entry) => entry.value === value)?.group === 'modality'
		) as ModalityKey[];
	}

	const TOGGLE_ITEM_CLASS =
		'bg-muted! border-border/30! shadow-none! dark:border-border/20! data-[state=on]:bg-muted-foreground/15! data-[state=on]:text-foreground! dark:data-[state=on]:bg-muted-foreground/25!';

	// none selected means every provider, so the label names the selection
	let providerLabel = $derived(
		providers.length === 0
			? 'All providers'
			: providers.length === 1
				? (backends.find((backend) => backend.id === providers[0])?.name ?? '1 provider')
				: `${providers.length} providers`
	);
	let contextLabel = $derived(
		CONTEXT_STEPS.find((step) => step.value === contextLimit)?.label ?? CONTEXT_STEPS[0].label
	);

	function toggleProvider(id: string, checked: boolean | 'indeterminate'): void {
		providers = checked === true ? [...providers, id] : providers.filter((entry) => entry !== id);
	}
</script>

{#snippet providerMark(backend: Backend)}
	{#if backend.id === LOCAL_BACKEND_ID}
		<BackendIcon {backend} class="h-3.5 w-3.5">
			{#snippet fallback()}
				<Logo class="shrink-0" style="--size: 0.875rem" />
			{/snippet}
		</BackendIcon>
	{:else}
		<BackendIcon {backend} class="h-3.5 w-3.5" />
	{/if}
{/snippet}

<ScrollCarousel alwaysShowArrows class="min-w-0 flex-1" gapSize="2" innerClass="items-center">
	{#if backends.length > 1}
		<DropdownMenu.Root>
			<DropdownMenu.Trigger>
				{#snippet child({ props })}
					<button
						{...props}
						class="inline-flex items-center whitespace-nowrap {FILTER_TRIGGER_CLASS}"
						type="button"
					>
						<Server class="h-3.5 w-3.5" />

						{providerLabel}

						<ChevronDown class="h-3.5 w-3.5 opacity-60" />
					</button>
				{/snippet}
			</DropdownMenu.Trigger>

			<DropdownMenu.Content align="start" class="min-w-48">
				<DropdownMenu.Group>
					<DropdownMenu.GroupHeading>Providers</DropdownMenu.GroupHeading>

					{#each backends as backend (backend.id)}
						<DropdownMenu.CheckboxItem
							checked={providers.includes(backend.id)}
							onCheckedChange={(checked) => toggleProvider(backend.id, checked)}
						>
							{@render providerMark(backend)}

							{backend.name}

							<DropdownMenu.Shortcut>{providerCounts[backend.id] ?? 0}</DropdownMenu.Shortcut>
						</DropdownMenu.CheckboxItem>
					{/each}
				</DropdownMenu.Group>

				{#if providers.length > 0}
					<DropdownMenu.Separator />

					<DropdownMenu.Item onSelect={() => (providers = [])}
						>Show every provider</DropdownMenu.Item
					>
				{/if}
			</DropdownMenu.Content>
		</DropdownMenu.Root>
	{/if}

	<Select.Root
		onValueChange={(value) => (contextLimit = Number(value))}
		type="single"
		value={String(contextLimit)}
	>
		<Select.Trigger class={FILTER_TRIGGER_CLASS} size="sm">
			<span class="text-muted-foreground">Context:</span>

			{contextLabel}
		</Select.Trigger>

		<Select.Content>
			{#each CONTEXT_STEPS as step (step.value)}
				<Select.Item label={step.label} value={String(step.value)}>{step.label}</Select.Item>
			{/each}
		</Select.Content>
	</Select.Root>

	<!-- a checkbox chip: the whole pill is the control, the box is its indicator -->
	<Toggle
		bind:pressed={draft}
		class="inline-flex items-center whitespace-nowrap {FILTER_TRIGGER_CLASS} px-2!"
		variant="outline"
	>
		<span
			aria-hidden="true"
			class="flex size-4 shrink-0 items-center justify-center rounded-[4px] border transition-shadow {draft
				? 'border-border/30 bg-muted-foreground/15 text-foreground dark:border-border/20 dark:bg-muted-foreground/25'
				: 'border-input bg-background dark:bg-input/30'}"
		>
			{#if draft}
				<Check class="size-3" />
			{/if}
		</span>

		Has draft sidecar
	</Toggle>

	<ToggleGroup.Root
		class="border border-border/30 bg-muted/60 shadow-sm dark:border-border/20 dark:bg-muted/75"
		onValueChange={setToggles}
		type="multiple"
		value={[...capabilities, ...modalities]}
		variant="outline"
	>
		{#each TOGGLES as toggle (toggle.value)}
			<ToggleGroup.Item
				aria-label={toggle.label}
				class={TOGGLE_ITEM_CLASS}
				title={toggle.label}
				value={toggle.value}
			>
				<toggle.icon class="h-3.5 w-3.5" />
			</ToggleGroup.Item>
		{/each}
	</ToggleGroup.Root>
</ScrollCarousel>
