<script lang="ts">
	import { Check, ChevronsUpDown } from '@lucide/svelte';
	import * as Command from '$lib/components/ui/command';
	import * as Popover from '$lib/components/ui/popover';
	import { Button } from '$lib/components/ui/button';
	import { cn } from '$lib/components/ui/utils';
	import { tick } from 'svelte';
	import { CUSTOM_WIDTH_PRESETS } from '$lib/utils/chat-width';

	interface Props {
		value: string;
		onChange: (value: string) => void;
		disabled?: boolean;
	}

	let { value = $bindable(''), onChange, disabled = false }: Props = $props();

	let open = $state(false);
	let triggerRef = $state<HTMLButtonElement>(null!);
    console.log("Rendering CustomWidthCombobox with value:", value);

	const widthPresets = Object.entries(CUSTOM_WIDTH_PRESETS).map(([key, pixelValue]) => ({
		value: key,
		label: `${key} (${pixelValue}px)`
	}));

	const displayValue = $derived.by(() => {
		const selectedWidthPreset = widthPresets.find((preset) => preset.value === value);
		if (selectedWidthPreset) return selectedWidthPreset.label;

		return 'Select width...';
	});

	function closeAndFocusTrigger() {
		open = false;
		tick().then(() => {
			triggerRef.focus();
		});
	}

	function handleSelectPreset(newValue: string) {
		value = newValue;
		onChange(newValue);
		closeAndFocusTrigger();
	}
</script>

<div class="flex w-full flex-col gap-2">
	<Popover.Root bind:open>
		<Popover.Trigger bind:ref={triggerRef}>
			{#snippet child({ props })}
				<Button
					{...props}
					variant="outline"
					class="w-full justify-between font-normal"
					role="combobox"
					aria-expanded={open}
					{disabled}
				>
					{displayValue}
					<ChevronsUpDown class="ml-2 h-4 w-4 shrink-0 opacity-50" />
				</Button>
			{/snippet}
		</Popover.Trigger>

		<Popover.Content class="z-[1000000] w-[var(--bits-popover-anchor-width)] p-0">
			<Command.Root>
				<Command.Input placeholder="Search presets" />
				<Command.List>
					<Command.Empty>No presets found.</Command.Empty>

					<Command.Group value="width-presets">
						{#each widthPresets as preset (preset.value)}
							<Command.Item value={preset.value} onSelect={() => handleSelectPreset(preset.value)}>
								<Check
									class={cn('mr-2 h-4 w-4', value === preset.value ? 'opacity-100' : 'opacity-0')}
								/>
								{preset.label}
							</Command.Item>
						{/each}
					</Command.Group>
				</Command.List>
			</Command.Root>
		</Popover.Content>
	</Popover.Root>
</div>
