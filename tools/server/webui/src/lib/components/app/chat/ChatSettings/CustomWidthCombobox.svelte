<script lang="ts">
	import { Check, ChevronsUpDown } from '@lucide/svelte';
	import * as Command from '$lib/components/ui/command';
	import * as Popover from '$lib/components/ui/popover';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { cn } from '$lib/components/ui/utils';
	import { tick } from 'svelte';
	import { CUSTOM_WIDTH_PRESETS } from '$lib/utils/chat-width';

	interface Props {
		value: string;
		onChange: (value: string) => void;
		disabled?: boolean;
	}

	let { value = $bindable(''), onChange, disabled = false }: Props = $props();
	console.log('Rendering CustomWidthCombobox with value:', value);

	let open = $state(false);
	let showCustomPixelInput = $state(false);
	let customPixelValue = $state('');
	let invalidCustomPixel = $state(false);

	let triggerRef = $state<HTMLButtonElement>(null!);
	let customPixelInputRef = $state<HTMLInputElement>(null!);

	const widthPresets = Object.entries(CUSTOM_WIDTH_PRESETS).map(([key, pixelValue]) => ({
		value: key,
		label: `${key} (${pixelValue}px)`
	}));

	const isValueANumber = $derived.by(() => {
		const numValue = Number(value);
		if (!isNaN(numValue) && numValue > 0) {
			return true;
		}
		return false;
	});

	const displayValue = $derived.by(() => {
		if (showCustomPixelInput) {
			return 'Set pixel value';
		}

		const selectedWidthPreset = widthPresets.find((preset) => preset.value === value);
		if (selectedWidthPreset) return selectedWidthPreset.label;

		if (isValueANumber) {
			return `Custom (${Number(value)}px)`;
		}

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
		showCustomPixelInput = false;
		onChange(newValue);
		closeAndFocusTrigger();
	}

	function handleShowCustomPixelInput() {
		open = false;
		showCustomPixelInput = true;

		tick().then(() => {
			if (customPixelInputRef) customPixelInputRef.focus();
		});
	}

	function handleCancelCustomPixelInput() {
		showCustomPixelInput = false;
		customPixelValue = '';
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			e.preventDefault();
			handleSaveCustomPixelInput();
		}
	}

	function handleSaveCustomPixelInput() {
		if (!customPixelValue) {
			invalidCustomPixel = true;
			return;
		}

		const number = Number(customPixelValue);
		const isValid = !isNaN(number) && number >= 300 && number <= 10000;

		if (isValid) {
			value = String(number);
			onChange(String(number));
			invalidCustomPixel = false;
			showCustomPixelInput = false;
		} else {
			invalidCustomPixel = true;
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
						<Command.Item value="set-custom-pixel-value" onSelect={handleShowCustomPixelInput}>
							<Check
								class={cn(
									'mr-2 h-4 w-4',
									showCustomPixelInput || isValueANumber ? 'opacity-100' : 'opacity-0'
								)}
							/>
							Set pixel value
						</Command.Item>

						<Command.Separator class="my-1" />

						{#each widthPresets as preset (preset.value)}
							<Command.Item value={preset.value} onSelect={() => handleSelectPreset(preset.value)}>
								<Check
									class={cn(
										'mr-2 h-4 w-4',
										value === preset.value && !showCustomPixelInput ? 'opacity-100' : 'opacity-0'
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

	{#if showCustomPixelInput}
		<div class="flex animate-in items-center gap-2 duration-200 fade-in slide-in-from-top-1">
			<Input
				bind:ref={customPixelInputRef}
				bind:value={customPixelValue}
				onkeydown={handleKeydown}
				type="number"
				placeholder="e.g. 800"
				class={cn(
					'flex-1',
					invalidCustomPixel && 'border-destructive focus-visible:ring-destructive'
				)}
				step="1"
				min="300"
				max="10000"
			/>
			<Button onclick={handleSaveCustomPixelInput}>Save</Button>
			<Button variant="outline" onclick={handleCancelCustomPixelInput}>Cancel</Button>
		</div>
		<p
			class={cn(
				'ml-1 text-[0.8rem]',
				invalidCustomPixel ? 'text-destructive' : 'text-muted-foreground'
			)}
		>
			Enter a value between 300 and 10000 pixels.
		</p>
	{/if}
</div>
