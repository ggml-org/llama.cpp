<script lang="ts">
	import type { ModelLoadProgress, ModelOverride } from '../utils';
	import { Info, ListOrdered } from '@lucide/svelte';
	import { CollapsibleSection } from '$lib/components/app';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Slider } from '$lib/components/ui/slider';
	import { Switch } from '$lib/components/ui/switch';
	import { LOAD_DEFAULTS, SPECULATIVE_OPTIONS } from '$lib/constants';
	import { formatParameters } from '$lib/utils/formatters';

	interface Props {
		contextMax: number;
		draft: ModelOverride;
		loadProgress: ModelLoadProgress | null;
		onChange: (draft: ModelOverride) => void;
		onReset: () => void;
		onSave: () => void;
	}

	let { contextMax, draft, loadProgress, onChange, onReset, onSave }: Props = $props();

	let load = $derived({ ...LOAD_DEFAULTS, ...draft.load });

	function patchLoad(patch: Partial<typeof load>): void {
		onChange({ ...draft, load: { ...draft.load, ...patch } });
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

<p class="pb-2 text-sm font-medium text-muted-foreground">Context and offload</p>

<div class="space-y-4">
	<div class="space-y-1.5">
		<div class="flex items-center gap-2">
			<span class="text-sm">Context Length</span>

			<span class="ml-auto">
				{@render numberInput(load.contextLength, (value) => patchLoad({ contextLength: value }))}
			</span>
		</div>

		<p class="text-xs text-muted-foreground">
			Model supports up to
			<Badge class="h-5 px-1.5 text-[10px]" variant="secondary">
				{formatParameters(contextMax)}
			</Badge>
			tokens
		</p>

		<Slider
			max={contextMax}
			onValueChange={(next: number) => patchLoad({ contextLength: next })}
			step={512}
			type="single"
			value={load.contextLength}
		/>
	</div>

	<div class="space-y-1.5">
		<div class="flex items-center gap-2">
			<span class="text-sm">
				GPU Offload <span class="text-muted-foreground">(layers)</span>
			</span>

			<span class="ml-auto">
				{@render numberInput(load.gpuOffload, (value) => patchLoad({ gpuOffload: value }))}
			</span>
		</div>

		<Slider
			max={200}
			onValueChange={(next: number) => patchLoad({ gpuOffload: next })}
			step={1}
			type="single"
			value={load.gpuOffload}
		/>
	</div>
</div>

<div class="mt-4">
	<CollapsibleSection triggerClass={sectionTrigger}>
		{#snippet trigger()}
			<ListOrdered class="h-3.5 w-3.5 text-muted-foreground" />

			<span class="text-sm font-medium">Advanced load params</span>
		{/snippet}

		<div class="pt-1">
			<div class={rowClass}>
				<span class="text-sm">CPU Thread Pool Size</span>

				<span class="ml-auto">
					{@render numberInput(load.cpuThreads, (value) => patchLoad({ cpuThreads: value }))}
				</span>
			</div>

			<div class={rowClass}>
				<span class="text-sm">Evaluation Batch Size</span>

				<span class="ml-auto">
					{@render numberInput(load.batchSize, (value) => patchLoad({ batchSize: value }))}
				</span>
			</div>

			<div class={rowClass}>
				<span class="text-sm">Physical Batch Size</span>

				<span class="ml-auto">
					{@render numberInput(load.ubatchSize, (value) => patchLoad({ ubatchSize: value }))}
				</span>
			</div>

			<div class={rowClass}>
				<span class="text-sm">Flash Attention</span>

				<Switch
					checked={load.flashAttention}
					class="ml-auto"
					onCheckedChange={(checked) => patchLoad({ flashAttention: checked === true })}
				/>
			</div>

			<div class={rowClass}>
				<span class="text-sm">Keep Model in Memory</span>

				<Switch
					checked={load.keepInMemory}
					class="ml-auto"
					onCheckedChange={(checked) => patchLoad({ keepInMemory: checked === true })}
				/>
			</div>

			<div class={rowClass}>
				<span class="text-sm">Try mmap()</span>

				<Switch
					checked={load.useMmap}
					class="ml-auto"
					onCheckedChange={(checked) => patchLoad({ useMmap: checked === true })}
				/>
			</div>

			<div class={rowClass}>
				<span class="text-sm">Speculative Decoding</span>

				<select
					class="ml-auto h-8 cursor-pointer rounded-md border border-input bg-transparent px-2 text-sm capitalize"
					onchange={(event) => patchLoad({ speculativeDecoding: event.currentTarget.value })}
					value={load.speculativeDecoding}
				>
					{#each SPECULATIVE_OPTIONS as value (value)}
						<option {value}>{value.replace('-', ' ')}</option>
					{/each}
				</select>
			</div>
		</div>
	</CollapsibleSection>
</div>

<div class="mt-4 flex items-start gap-2 rounded-lg bg-muted/40 p-3 text-xs text-muted-foreground">
	<Info class="mt-0.5 h-3.5 w-3.5 shrink-0" />

	<p>Changes apply on next load - eject and re-load to pick them up.</p>
</div>

{#if loadProgress}
	<p class="pt-2 text-xs text-muted-foreground">
		Loading: {loadProgress.current}
		{Math.round(loadProgress.value * 100)}%
	</p>
{/if}

<div class="flex justify-end gap-2 pt-4">
	<Button onclick={onReset} variant="ghost">Reset</Button>

	<Button onclick={onSave}>Save</Button>
</div>
