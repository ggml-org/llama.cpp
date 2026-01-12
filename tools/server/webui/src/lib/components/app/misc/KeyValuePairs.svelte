<script lang="ts">
	import { Plus, X } from '@lucide/svelte';
	import { Input } from '$lib/components/ui/input';
	import { autoResizeTextarea } from '$lib/utils';
	import type { KeyValuePair } from '$lib/types';

	interface Props {
		pairs: KeyValuePair[];
		onPairsChange: (pairs: KeyValuePair[]) => void;
		keyPlaceholder?: string;
		valuePlaceholder?: string;
		addButtonLabel?: string;
		emptyMessage?: string;
		sectionLabel?: string;
		sectionLabelOptional?: boolean;
	}

	let {
		pairs,
		onPairsChange,
		keyPlaceholder = 'Key',
		valuePlaceholder = 'Value',
		addButtonLabel = 'Add',
		emptyMessage = 'No items configured.',
		sectionLabel,
		sectionLabelOptional = true
	}: Props = $props();

	function addPair() {
		onPairsChange([...pairs, { key: '', value: '' }]);
	}

	function removePair(index: number) {
		onPairsChange(pairs.filter((_, i) => i !== index));
	}

	function updatePairKey(index: number, key: string) {
		const newPairs = [...pairs];
		newPairs[index] = { ...newPairs[index], key };
		onPairsChange(newPairs);
	}

	function updatePairValue(index: number, value: string) {
		const newPairs = [...pairs];
		newPairs[index] = { ...newPairs[index], value };
		onPairsChange(newPairs);
	}
</script>

<div>
	<div class="mb-1 flex items-center justify-between">
		{#if sectionLabel}
			<span class="text-xs font-medium">
				{sectionLabel}
				{#if sectionLabelOptional}
					<span class="text-muted-foreground">(optional)</span>
				{/if}
			</span>
		{/if}

		<button
			type="button"
			class="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
			onclick={addPair}
		>
			<Plus class="h-3 w-3" />
			{addButtonLabel}
		</button>
	</div>
	{#if pairs.length > 0}
		<div class="space-y-2">
			{#each pairs as pair, index (index)}
				<div class="flex items-start gap-2">
					<Input
						type="text"
						placeholder={keyPlaceholder}
						value={pair.key}
						oninput={(e) => updatePairKey(index, e.currentTarget.value)}
						class="flex-1"
					/>
					<textarea
						placeholder={valuePlaceholder}
						value={pair.value}
						oninput={(e) => {
							updatePairValue(index, e.currentTarget.value);
							autoResizeTextarea(e.currentTarget);
						}}
						class="flex-1 resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm leading-5 placeholder:text-muted-foreground focus-visible:ring-1 focus-visible:ring-ring focus-visible:outline-none"
						rows="1"
					></textarea>
					<button
						type="button"
						class="shrink-0 p-1 text-muted-foreground hover:text-destructive"
						onclick={() => removePair(index)}
						aria-label="Remove item"
					>
						<X class="h-3.5 w-3.5" />
					</button>
				</div>
			{/each}
		</div>
	{:else}
		<p class="text-xs text-muted-foreground">{emptyMessage}</p>
	{/if}
</div>
