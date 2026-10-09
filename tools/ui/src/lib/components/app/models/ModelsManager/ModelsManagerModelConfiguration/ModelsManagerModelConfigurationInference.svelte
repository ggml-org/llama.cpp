<script lang="ts">
	import type { ModelOverride } from '../utils';
	import { Braces, CircleDot, SlidersHorizontal, X } from '@lucide/svelte';
	import { CollapsibleSection } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Input } from '$lib/components/ui/input';
	import { Slider } from '$lib/components/ui/slider';
	import { Switch } from '$lib/components/ui/switch';
	import { Textarea } from '$lib/components/ui/textarea';
	import { SAMPLING_DEFAULTS } from '$lib/constants';

	interface Props {
		draft: ModelOverride;
		onChange: (draft: ModelOverride) => void;
		onReset: () => void;
		onSave: () => void;
	}

	let { draft, onChange, onReset, onSave }: Props = $props();

	let stopDraft = $state('');
	let stopStrings = $derived(draft.stopStrings ?? []);

	function patchSampling(patch: Partial<NonNullable<ModelOverride['sampling']>>): void {
		onChange({ ...draft, sampling: { ...draft.sampling, ...patch } });
	}

	function addStopString(): void {
		const value = stopDraft.trim();

		if (!value) return;

		onChange({ ...draft, stopStrings: [...stopStrings, value] });
		stopDraft = '';
	}

	function removeStopString(value: string): void {
		onChange({ ...draft, stopStrings: stopStrings.filter((s) => s !== value) });
	}

	const rowClass = 'flex items-center gap-3 py-2';
	const sectionTrigger = 'flex w-full cursor-pointer items-center gap-2 py-2 text-left';
</script>

{#snippet numberInput(value: number, onInput: (value: number) => void)}
	<Input
		class="h-8 w-24 text-right text-sm"
		min="0"
		oninput={(event) => onInput(Number(event.currentTarget.value) || 0)}
		type="number"
		{value}
	/>
{/snippet}

{#snippet samplingRow(
	label: string,
	value: number | null | undefined,
	enabled: boolean,
	onToggle: (enabled: boolean) => void,
	onValue: (value: number) => void,
	max: number,
	step: number
)}
	<div class="space-y-1.5 py-2">
		<div class="flex items-center gap-2">
			<span class="text-sm">{label}</span>

			<Checkbox checked={enabled} onCheckedChange={(checked) => onToggle(checked === true)} />

			<span class="ml-auto">
				{@render numberInput(value ?? 0, onValue)}
			</span>
		</div>

		<Slider
			disabled={!enabled}
			{max}
			onValueChange={(next: number) => onValue(next)}
			{step}
			type="single"
			value={value ?? 0}
		/>
	</div>
{/snippet}

<p class="pb-2 text-sm font-medium text-muted-foreground">System prompt</p>

<Textarea
	class="min-h-24 text-sm"
	oninput={(event) => onChange({ ...draft, systemPrompt: event.currentTarget.value })}
	placeholder="Example, &quot;Only answer in rhymes&quot;"
	value={draft.systemPrompt ?? ''}
/>

<p class="pt-1 text-right text-xs text-muted-foreground">Token count: N/A</p>

<div class="mt-4">
	<CollapsibleSection triggerClass={sectionTrigger}>
		{#snippet trigger()}
			<SlidersHorizontal class="h-3.5 w-3.5 text-muted-foreground" />

			<span class="text-sm font-medium">Sampling</span>
		{/snippet}

		<div class="pt-1">
			{@render samplingRow(
				'Temperature',
				draft.sampling?.temperature,
				draft.sampling?.temperature !== null && draft.sampling?.temperature !== undefined,
				(enabled) => patchSampling({ temperature: enabled ? SAMPLING_DEFAULTS.temperature : null }),
				(value) => patchSampling({ temperature: value }),
				2,
				0.05
			)}

			{@render samplingRow(
				'Top K',
				draft.sampling?.topK,
				draft.sampling?.topK !== null && draft.sampling?.topK !== undefined,
				(enabled) => patchSampling({ topK: enabled ? SAMPLING_DEFAULTS.topK : null }),
				(value) => patchSampling({ topK: value }),
				200,
				1
			)}

			{@render samplingRow(
				'Top P',
				draft.sampling?.topP,
				draft.sampling?.topP !== null && draft.sampling?.topP !== undefined,
				(enabled) => patchSampling({ topP: enabled ? SAMPLING_DEFAULTS.topP : null }),
				(value) => patchSampling({ topP: value }),
				1,
				0.01
			)}

			{@render samplingRow(
				'Min P',
				draft.sampling?.minP,
				draft.sampling?.minP !== null && draft.sampling?.minP !== undefined,
				(enabled) => patchSampling({ minP: enabled ? SAMPLING_DEFAULTS.minP : null }),
				(value) => patchSampling({ minP: value }),
				1,
				0.01
			)}

			{@render samplingRow(
				'Repeat Penalty',
				draft.sampling?.repeatPenalty,
				draft.sampling?.repeatPenalty !== null && draft.sampling?.repeatPenalty !== undefined,
				(enabled) =>
					patchSampling({ repeatPenalty: enabled ? SAMPLING_DEFAULTS.repeatPenalty : null }),
				(value) => patchSampling({ repeatPenalty: value }),
				2,
				0.01
			)}
		</div>
	</CollapsibleSection>

	<CollapsibleSection triggerClass={sectionTrigger}>
		{#snippet trigger()}
			<span class="text-sm font-medium">Stop strings</span>
		{/snippet}

		<div class="space-y-2 pt-1">
			{#if stopStrings.length > 0}
				<div class="flex flex-wrap gap-1.5">
					{#each stopStrings as value (value)}
						<button
							class="inline-flex cursor-pointer items-center gap-1 rounded-full border border-border px-2 py-0.5 text-xs"
							onclick={() => removeStopString(value)}
							type="button"
						>
							{value}

							<X class="h-3 w-3" />
						</button>
					{/each}
				</div>
			{/if}

			<Input
				bind:value={stopDraft}
				class="h-9 text-sm"
				onkeydown={(event) => {
					if (event.key === 'Enter') {
						event.preventDefault();
						addStopString();
					}
				}}
				placeholder="Enter a string and press Enter"
			/>
		</div>
	</CollapsibleSection>

	<CollapsibleSection triggerClass={sectionTrigger}>
		{#snippet trigger()}
			<Braces class="h-3.5 w-3.5 text-muted-foreground" />

			<span class="text-sm font-medium">Structured output</span>
		{/snippet}

		<div class="space-y-2 pt-1">
			<div class={rowClass}>
				<span class="text-sm">Enabled</span>

				<Switch
					checked={draft.structuredOutput?.enabled ?? false}
					class="ml-auto"
					onCheckedChange={(checked) =>
						onChange({
							...draft,
							structuredOutput: {
								enabled: checked === true,
								schema: draft.structuredOutput?.schema ?? ''
							}
						})}
				/>
			</div>

			<Textarea
				class="min-h-20 font-mono text-xs"
				disabled={!draft.structuredOutput?.enabled}
				oninput={(event) =>
					onChange({
						...draft,
						structuredOutput: {
							enabled: draft.structuredOutput?.enabled ?? false,
							schema: event.currentTarget.value
						}
					})}
				placeholder={'{ }'}
				value={draft.structuredOutput?.schema ?? ''}
			/>
		</div>
	</CollapsibleSection>

	<CollapsibleSection triggerClass={sectionTrigger}>
		{#snippet trigger()}
			<CircleDot class="h-3.5 w-3.5 text-muted-foreground" />

			<span class="text-sm font-medium">Reasoning</span>
		{/snippet}

		<div class="space-y-2 pt-1">
			<div class={rowClass}>
				<span class="text-sm">Enable Thinking</span>

				<Switch
					checked={draft.reasoning?.enabled ?? false}
					class="ml-auto"
					onCheckedChange={(checked) =>
						onChange({
							...draft,
							reasoning: {
								budget: draft.reasoning?.budget ?? 'Unrestricted',
								enabled: checked === true
							}
						})}
				/>
			</div>

			<p class="text-xs text-muted-foreground">
				Controls whether the model will think before replying
			</p>

			<div class={rowClass}>
				<span class="text-sm">Reasoning Budget</span>

				<span class="ml-auto text-sm text-muted-foreground">
					{draft.reasoning?.budget ?? 'Unrestricted'}
				</span>
			</div>
		</div>
	</CollapsibleSection>
</div>

<div class="flex justify-end gap-2 pt-4">
	<Button onclick={onReset} variant="ghost">Reset</Button>

	<Button onclick={onSave}>Save</Button>
</div>
