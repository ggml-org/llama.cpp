<script lang="ts">
	import { Check, ChevronsUpDown } from '@lucide/svelte';
	import * as Command from '$lib/components/ui/command';
	import * as Popover from '$lib/components/ui/popover';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { cn } from '$lib/components/ui/utils';
	import { tick } from 'svelte';
	import {
		CUSTOM_WIDTH_PRESETS,
		MIN_CUSTOM_WIDTH,
		MAX_CUSTOM_WIDTH
	} from '$lib/constants/chat-width';

	interface Props {
		value: string;
		onChange: (value: string) => void;
		disabled?: boolean;
	}

	let { value = $bindable(''), onChange, disabled = false }: Props = $props();

	let open = $state(false);
	let isEditing = $state(false);
	let inputValue = $state('');
	let isError = $state(false);

	let triggerRef = $state<HTMLButtonElement>(null!);
	let inputRef = $state<HTMLInputElement>(null!);

	const presets = Object.entries(CUSTOM_WIDTH_PRESETS).map(([key, pixelValue]) => ({
		value: key,
		label: `${key} (${pixelValue}px)`
	}));

	const displayLabel = $derived.by(() => {
		if (isEditing) return 'Set pixel value';

		const foundPreset = presets.find((p) => p.value === value);
		if (foundPreset) return foundPreset.label;

		const number = Number(value);
		if (!isNaN(number) && number > 0) {
			return `Custom (${number}px)`;
		}

		return 'Select width...';
	});

	function closePopover() {
		open = false;
		tick().then(() => triggerRef?.focus());
	}

	function selectPreset(newValue: string) {
		value = newValue;
		isEditing = false;
		isError = false;
		onChange(newValue);
		closePopover();
	}

	function startEditing() {
		open = false;
		isEditing = true;
		inputValue = '';
		isError = false;

		tick().then(() => inputRef?.focus());
	}

	function cancelEditing() {
		isEditing = false;
		inputValue = '';
		isError = false;
	}

	function submitCustomValue() {
		if (!inputValue) {
			isError = true;
			return;
		}

		const number = Number(inputValue);
		const isValid = !isNaN(number) && number >= MIN_CUSTOM_WIDTH && number <= MAX_CUSTOM_WIDTH;

		if (isValid) {
			value = String(number);
			onChange(String(number));
			isEditing = false;
			isError = false;
		} else {
			isError = true;
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			e.preventDefault();
			submitCustomValue();
		}
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
					{displayLabel}
					<ChevronsUpDown class="ml-2 h-4 w-4 shrink-0 opacity-50" />
				</Button>
			{/snippet}
		</Popover.Trigger>

		<Popover.Content class="z-[1000000] w-[var(--bits-popover-anchor-width)] p-0">
			<Command.Root>
				<Command.Input placeholder="Search presets" />
				<Command.List>
					<Command.Empty>No presets found.</Command.Empty>

					<Command.Group>
						<Command.Item value="custom-input-trigger" onSelect={startEditing}>
							<Check class={cn('mr-2 h-4 w-4', isEditing ? 'opacity-100' : 'opacity-0')} />
							Set pixel value
						</Command.Item>

						<Command.Separator class="my-1" />

						{#each presets as preset (preset.value)}
							<Command.Item value={preset.value} onSelect={() => selectPreset(preset.value)}>
								<Check
									class={cn(
										'mr-2 h-4 w-4',
										value === preset.value && !isEditing ? 'opacity-100' : 'opacity-0'
									)}
								/>
								{preset.label}
							</Command.Item>
						{/each}
					</Command.Group>
				</Command.List>
			</Command.Root>
		</Popover.Content>
	</Popover.Root>

	{#if isEditing}
		<div class="flex animate-in items-start gap-2 duration-200 fade-in slide-in-from-top-1">
			<div class="flex-1 space-y-1">
				<Input
					bind:ref={inputRef}
					bind:value={inputValue}
					onkeydown={handleKeydown}
					type="number"
					placeholder="e.g. 800"
					class={cn(isError && 'border-destructive focus-visible:ring-destructive')}
					min={MIN_CUSTOM_WIDTH}
					max={MAX_CUSTOM_WIDTH}
				/>
				<p class={cn('text-[0.8rem]', isError ? 'text-destructive' : 'text-muted-foreground')}>
					Enter a value between {MIN_CUSTOM_WIDTH} and {MAX_CUSTOM_WIDTH}.
				</p>
			</div>

			<div class="flex gap-2">
				<Button size="sm" onclick={submitCustomValue}>Save</Button>
				<Button size="sm" variant="outline" onclick={cancelEditing}>Cancel</Button>
			</div>
		</div>
	{/if}
</div>
