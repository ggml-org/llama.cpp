<script lang="ts">
	import { ScrollCarousel } from '$lib/components/app';
	import * as Select from '$lib/components/ui/select';
	import * as ToggleGroup from '$lib/components/ui/toggle-group';
	import {
		CAPABILITY_ICONS,
		CAPABILITY_LABELS,
		FILTER_TOGGLE_ITEM_CLASS,
		FILTER_TRIGGER_CLASS,
		MODALITY_FLAG_KEYS,
		MODALITY_ICONS,
		MODALITY_KEYS,
		MODALITY_LABELS,
		MODALITY_ORDER,
		type ModalityKey
	} from '$lib/constants';
	import { ModelCapability } from '$lib/enums';
	import { modelsStore } from '$lib/stores';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		/** Capabilities a model must have every one of. */
		capabilities?: ModelCapability[];
		/** Smallest context a model must support; 0 keeps every model. */
		contextLimit?: number;
		/** Modalities a model must support at least one of. */
		modalities?: ModalityKey[];
	}

	let {
		capabilities = $bindable<ModelCapability[]>([]),
		contextLimit = $bindable(0),
		modalities = $bindable<ModalityKey[]>([])
	}: Props = $props();

	const CONTEXT_STEPS: { label: string; value: number }[] = [
		{ label: 'Any context', value: 0 },
		{ label: '8K or more', value: 8192 },
		{ label: '32K or more', value: 32768 },
		{ label: '128K or more', value: 131_072 },
		{ label: '256K or more', value: 262_144 },
		{ label: '1M or more', value: 1_048_576 }
	];

	const CAPABILITY_TOGGLES = [ModelCapability.TOOL_USE, ModelCapability.REASONING].map((value) => ({
		icon: CAPABILITY_ICONS[value],
		label: CAPABILITY_LABELS[value],
		value
	}));

	// only the modalities the models at hand actually carry: a toggle for something
	// none of them supports could only ever empty the table
	let detectedModalities = $derived.by(() => {
		const keys = new SvelteSet<ModalityKey>();

		for (const option of modelsStore.models) {
			for (const key of MODALITY_KEYS) {
				if (option.modalities?.[key]) keys.add(key);
			}
		}

		return keys;
	});
	let modalityToggles = $derived(
		MODALITY_ORDER.filter((modality) => detectedModalities.has(MODALITY_FLAG_KEYS[modality])).map(
			(modality) => ({
				icon: MODALITY_ICONS[modality],
				label: MODALITY_LABELS[modality],
				value: MODALITY_FLAG_KEYS[modality]
			})
		)
	);

	// one group holds what a model can do and what it can accept
	let toggles = $derived([...CAPABILITY_TOGGLES, ...modalityToggles]);
	const CAPABILITY_VALUES = new Set<string>(CAPABILITY_TOGGLES.map((entry) => entry.value));
	// every flag value, so a choice made before a model disappeared still round-trips
	const MODALITY_VALUES = new Set<string>(MODALITY_KEYS);

	// the group holds one flat list, so a change splits back into the two filters
	function setToggles(values: string[]): void {
		capabilities = values.filter((value): value is ModelCapability => CAPABILITY_VALUES.has(value));
		modalities = values.filter((value): value is ModalityKey => MODALITY_VALUES.has(value));
	}

	let contextLabel = $derived(
		CONTEXT_STEPS.find((step) => step.value === contextLimit)?.label ?? CONTEXT_STEPS[0].label
	);
</script>

<!-- below md the carousel keeps a row of its own: a toolbar that also holds a call to
     action would otherwise squeeze the filters out of sight -->
<ScrollCarousel
	alwaysShowArrows
	class="min-w-0 flex-1 max-md:w-full max-md:flex-none"
	gapSize="2"
	innerClass="items-center"
>
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

	<ToggleGroup.Root
		class="border border-border/30 bg-muted/60 shadow-sm dark:border-border/20 dark:bg-muted/75"
		onValueChange={setToggles}
		type="multiple"
		value={[...capabilities, ...modalities]}
		variant="outline"
	>
		{#each toggles as toggle (toggle.value)}
			<ToggleGroup.Item
				aria-label={toggle.label}
				class={FILTER_TOGGLE_ITEM_CLASS}
				title={toggle.label}
				value={toggle.value}
			>
				<toggle.icon class="h-3.5 w-3.5" />
			</ToggleGroup.Item>
		{/each}
	</ToggleGroup.Root>
</ScrollCarousel>
