<script lang="ts">
	import {
		isLocalOption,
		LOAD_DEFAULTS,
		type ModelOverride,
		modelQuantLabel,
		modelSizeLabel,
		SAMPLING_DEFAULTS,
		SPECULATIVE_OPTIONS
	} from './utils';
	import {
		Braces,
		CircleDot,
		Info,
		ListOrdered,
		Play,
		SlidersHorizontal,
		Square,
		X
	} from '@lucide/svelte';
	import { CollapsibleSection } from '$lib/components/app';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Input } from '$lib/components/ui/input';
	import { Slider } from '$lib/components/ui/slider';
	import { Switch } from '$lib/components/ui/switch';
	import * as Tabs from '$lib/components/ui/tabs';
	import { Textarea } from '$lib/components/ui/textarea';
	import { ServerModelStatus } from '$lib/enums';
	import { modelsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { formatParameters } from '$lib/utils/formatters';

	interface Props {
		isCustomized: boolean;
		onClose: () => void;
		onSave: (override: ModelOverride) => void;
		onToggleLoad: () => void;
		onUseInNewChat: () => void;
		option: ModelOption;
		override?: ModelOverride;
	}

	let { isCustomized, onClose, onSave, onToggleLoad, onUseInNewChat, option, override }: Props =
		$props();

	let tab = $state('info');
	let draft = $state<ModelOverride>({});
	let stopDraft = $state('');

	// The draft follows the stored override of whichever model is picked
	$effect(() => {
		draft = override ?? {};
		stopDraft = '';
	});

	let serverProps = $derived(modelsStore.props.getModelProps(option.model));
	let status = $derived.by(() => {
		const model = modelsStore.routerModels.find((m) => m.id === option.model);

		return (model?.status?.value as ServerModelStatus) ?? null;
	});
	let isOperationInProgress = $derived(modelsStore.status.isOperationInProgress(option.model));
	let isLoaded = $derived(
		(status === ServerModelStatus.LOADED || status === ServerModelStatus.SLEEPING) &&
			!isOperationInProgress
	);
	let loadProgress = $derived(
		isOperationInProgress ? modelsStore.status.getLoadProgress(option.model) : null
	);

	let load = $derived({ ...LOAD_DEFAULTS, ...draft.load });
	let contextMax = $derived(
		option.contextLength ??
			serverProps?.default_generation_settings?.n_ctx ??
			LOAD_DEFAULTS.contextLength
	);
	let quant = $derived(modelQuantLabel(option));
	let size = $derived(modelSizeLabel(option));
	let stopStrings = $derived(draft.stopStrings ?? []);
	let infoRows = $derived([
		{ label: 'Model', value: option.model },
		{ label: 'File Path', value: serverProps?.model_path ?? null },
		{
			label: 'Context Size',
			value: serverProps
				? `${formatParameters(serverProps.default_generation_settings.n_ctx)} tokens`
				: null
		},
		{
			label: 'Training Context',
			value: option.contextLength ? `${formatParameters(option.contextLength)} tokens` : null
		},
		{ label: 'Model Size', value: size },
		{ label: 'Parameters', value: option.parsedId?.params ?? null },
		{ isBadge: true, label: 'Quantization', value: quant },
		{ isBadge: true, label: 'Architecture', value: (option.meta?.architecture as string) ?? null },
		{ label: 'Format', value: isLocalOption(option) ? 'GGUF' : null },
		{ label: 'Parallel Slots', value: serverProps ? String(serverProps.total_slots) : null }
	] satisfies Array<{ isBadge?: boolean; label: string; value: string | null }>);

	function patchLoad(patch: Partial<typeof load>): void {
		draft = { ...draft, load: { ...draft.load, ...patch } };
	}

	function patchSampling(patch: Partial<NonNullable<ModelOverride['sampling']>>): void {
		draft = { ...draft, sampling: { ...draft.sampling, ...patch } };
	}

	function addStopString(): void {
		const value = stopDraft.trim();

		if (!value) return;

		draft = { ...draft, stopStrings: [...stopStrings, value] };
		stopDraft = '';
	}

	function removeStopString(value: string): void {
		draft = { ...draft, stopStrings: stopStrings.filter((s) => s !== value) };
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

<div class="flex h-full min-h-0 flex-col">
	<div class="flex items-start gap-3 px-4 pt-4">
		<span
			class="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-muted text-xs font-medium text-muted-foreground"
		>
			{option.parsedId?.params ?? '—'}
		</span>

		<div class="min-w-0 flex-1">
			<div class="flex items-center gap-2">
				<span class="truncate text-base font-medium">{option.name}</span>

				{#if isCustomized}
					<Badge class="h-5 px-1.5 text-[10px]" variant="secondary">CUSTOMIZED</Badge>
				{/if}
			</div>

			<p class="truncate text-xs text-muted-foreground">
				{option.parsedId?.orgName ?? 'unknown'} · {isLoaded ? 'loaded' : 'not loaded'}{size
					? ` · ${size}`
					: ''}
			</p>
		</div>

		<Button
			aria-label="Close details"
			class="h-7 w-7"
			onclick={onClose}
			size="icon"
			variant="ghost"
		>
			<X class="h-4 w-4" />
		</Button>
	</div>

	<div class="flex gap-2 px-4 pt-3">
		<Button class="flex-1 gap-1.5" onclick={onUseInNewChat} variant="outline">
			<Play class="h-3.5 w-3.5" />

			Use in New Chat
		</Button>

		<Button class="flex-1 gap-1.5" onclick={onToggleLoad} variant="outline">
			<Square class="h-3.5 w-3.5" />

			{isLoaded ? 'Eject Model' : 'Load Model'}
		</Button>
	</div>

	<Tabs.Root class="mt-3 min-h-0 flex-1 gap-0" onValueChange={(value) => (tab = value)} value={tab}>
		<Tabs.List class="px-4">
			<Tabs.Trigger value="info">Info</Tabs.Trigger>

			<Tabs.Trigger value="load">Load</Tabs.Trigger>

			<Tabs.Trigger value="inference">Inference</Tabs.Trigger>
		</Tabs.List>

		<div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
			<Tabs.Content value="info">
				<div class="space-y-0">
					{#each infoRows as row (row.label)}
						<div class="flex items-center gap-3 border-b border-border/30 py-2.5 last:border-b-0">
							<span class="text-sm text-muted-foreground">{row.label}</span>

							<span class="ml-auto min-w-0 truncate text-sm">
								{#if row.value === null}
									<span class="text-muted-foreground">—</span>
								{:else if row.isBadge}
									<Badge class="h-5 px-1.5 text-[10px]" variant="secondary">{row.value}</Badge>
								{:else}
									{row.value}
								{/if}
							</span>
						</div>
					{/each}
				</div>

				<div class="mt-4">
					<CollapsibleSection triggerClass={sectionTrigger}>
						{#snippet trigger()}
							<span class="text-sm font-medium">Chat Template</span>
						{/snippet}

						<pre
							class="mt-1 max-h-48 overflow-auto rounded-md bg-muted/50 p-2 text-xs whitespace-pre-wrap">{serverProps?.chat_template ??
								'Not reported by the server.'}</pre>
					</CollapsibleSection>

					<CollapsibleSection triggerClass={sectionTrigger}>
						{#snippet trigger()}
							<span class="text-sm font-medium">Source File</span>
						{/snippet}

						<div class="space-y-2 pt-1">
							<p class="text-xs break-all text-muted-foreground">
								{serverProps?.model_path ?? 'Path is reported once the model is loaded.'}
							</p>
						</div>
					</CollapsibleSection>
				</div>
			</Tabs.Content>

			<Tabs.Content value="load">
				<p class="pb-2 text-sm font-medium text-muted-foreground">Context and offload</p>

				<div class="space-y-4">
					<div class="space-y-1.5">
						<div class="flex items-center gap-2">
							<span class="text-sm">Context Length</span>

							<span class="ml-auto">
								{@render numberInput(load.contextLength, (value) =>
									patchLoad({ contextLength: value })
								)}
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
									{@render numberInput(load.cpuThreads, (value) =>
										patchLoad({ cpuThreads: value })
									)}
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
									{@render numberInput(load.ubatchSize, (value) =>
										patchLoad({ ubatchSize: value })
									)}
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
									onchange={(event) =>
										patchLoad({ speculativeDecoding: event.currentTarget.value })}
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

				<div
					class="mt-4 flex items-start gap-2 rounded-lg bg-muted/40 p-3 text-xs text-muted-foreground"
				>
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
					<Button onclick={() => (draft = override ?? {})} variant="ghost">Reset</Button>

					<Button onclick={() => onSave(draft)}>Save</Button>
				</div>
			</Tabs.Content>

			<Tabs.Content value="inference">
				<p class="pb-2 text-sm font-medium text-muted-foreground">System prompt</p>

				<Textarea
					class="min-h-24 text-sm"
					oninput={(event) => (draft = { ...draft, systemPrompt: event.currentTarget.value })}
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
								(enabled) =>
									patchSampling({ temperature: enabled ? SAMPLING_DEFAULTS.temperature : null }),
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
								draft.sampling?.repeatPenalty !== null &&
									draft.sampling?.repeatPenalty !== undefined,
								(enabled) =>
									patchSampling({
										repeatPenalty: enabled ? SAMPLING_DEFAULTS.repeatPenalty : null
									}),
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
										(draft = {
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
									(draft = {
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
										(draft = {
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
					<Button onclick={() => (draft = override ?? {})} variant="ghost">Reset</Button>

					<Button onclick={() => onSave(draft)}>Save</Button>
				</div>
			</Tabs.Content>
		</div>
	</Tabs.Root>
</div>
