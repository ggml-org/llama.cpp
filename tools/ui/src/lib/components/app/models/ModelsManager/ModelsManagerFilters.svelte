<script lang="ts">
	import { MODALITY_KEYS, type ModalityKey } from './utils';
	import { Image, Lightbulb, Mic, Video, Wrench } from '@lucide/svelte';
	import { ScrollCarousel } from '$lib/components/app';
	import * as Select from '$lib/components/ui/select';
	import * as ToggleGroup from '$lib/components/ui/toggle-group';
	import { FILTER_TRIGGER_CLASS } from '$lib/constants';
	import { ModelCapability } from '$lib/enums';

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

	const CAPABILITY_TOGGLES: { icon: typeof Wrench; label: string; value: ModelCapability }[] = [
		{ icon: Wrench, label: 'Tool use', value: ModelCapability.TOOL_USE },
		{ icon: Lightbulb, label: 'Reasoning', value: ModelCapability.REASONING }
	];

	const MODALITY_TOGGLES: { icon: typeof Wrench; label: string; value: ModalityKey }[] = [
		{ icon: Image, label: 'Vision', value: 'vision' },
		{ icon: Video, label: 'Video', value: 'video' },
		{ icon: Mic, label: 'Audio', value: 'audio' }
	];

	// one group holds everything a model either has or does not: what it can do,
	// and what it can accept
	const TOGGLES = [...CAPABILITY_TOGGLES, ...MODALITY_TOGGLES];
	const CAPABILITY_VALUES = new Set<string>(CAPABILITY_TOGGLES.map((entry) => entry.value));

	// the group holds one flat list, so a change splits back into the two filters
	function setToggles(values: string[]): void {
		capabilities = values.filter((value): value is ModelCapability => CAPABILITY_VALUES.has(value));
		modalities = values.filter(
			(value): value is ModalityKey =>
				!CAPABILITY_VALUES.has(value) && (MODALITY_KEYS as string[]).includes(value)
		);
	}

	const TOGGLE_ITEM_CLASS =
		'bg-muted! border-border/30! shadow-none! dark:border-border/20! data-[state=on]:bg-muted-foreground/15! data-[state=on]:text-foreground! dark:data-[state=on]:bg-muted-foreground/25!';

	let contextLabel = $derived(
		CONTEXT_STEPS.find((step) => step.value === contextLimit)?.label ?? CONTEXT_STEPS[0].label
	);
</script>

<ScrollCarousel alwaysShowArrows class="min-w-0 flex-1" gapSize="2" innerClass="items-center">
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
