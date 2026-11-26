<!--
	A select whose trigger looks like ours but opens the OS's own picker: a native <select> lies
	invisible over the trigger and takes the tap, so a phone shows its wheel or sheet.
-->
<script lang="ts">
	import { ChevronDown, ChevronsUpDown } from '@lucide/svelte';
	import { NATIVE_SELECT_PLACEHOLDER } from '../../../constants';

	interface NativeSelectOption {
		label: string;
		value: string;
	}

	interface Props {
		/** Shown on the trigger in place of the selected option's label. */
		displayLabel?: string;
		disabled?: boolean;
		/** Options under a heading of their own, after the ungrouped ones. */
		groups?: Array<{ heading: string; options: NativeSelectOption[] }>;
		id?: string;
		label: string;
		onValueChange: (value: string) => void;
		options?: NativeSelectOption[];
		placeholder?: string;
		value: string;
		/** `default` is the boxed select trigger; `inset` is a value and chevron at a row's end. */
		variant?: 'default' | 'inset';
		class?: string;
	}

	let {
		class: className = '',
		disabled = false,
		displayLabel,
		groups = [],
		id,
		label,
		onValueChange,
		options = [],
		placeholder = NATIVE_SELECT_PLACEHOLDER,
		value,
		variant = 'default'
	}: Props = $props();

	let selected = $derived(
		[...options, ...groups.flatMap((group) => group.options)].find(
			(option) => option.value === value
		)
	);
	let shown = $derived(displayLabel ?? selected?.label ?? placeholder);
</script>

<div
	class={[
		'relative flex min-w-0 items-center has-[select:focus-visible]:ring-3 has-[select:focus-visible]:ring-ring/50',
		variant === 'inset'
			? 'max-w-[60%] gap-1 rounded-md text-body-sm text-muted-foreground'
			: 'h-9 w-full justify-between gap-2 rounded-md border border-border/30 bg-muted/60 px-3 text-body-sm matte-key dark:border-border/20 dark:bg-muted/75',
		disabled && 'opacity-50',
		className
	]}
>
	<span class={['truncate', !selected && !displayLabel && 'text-muted-foreground']}>{shown}</span>

	{#if variant === 'inset'}
		<ChevronsUpDown class="size-3.5 shrink-0" />
	{:else}
		<ChevronDown class="size-3.5 shrink-0 opacity-60" />
	{/if}

	<!-- 16px, or iOS zooms the page in when the picker opens -->
	<select
		aria-label={label}
		class="absolute inset-0 size-full cursor-pointer appearance-none text-body opacity-0 disabled:cursor-not-allowed"
		{disabled}
		{id}
		onchange={(event) => onValueChange(event.currentTarget.value)}
		{value}
	>
		{#if !selected}
			<option disabled {value}>{placeholder}</option>
		{/if}

		{#each options as option (option.value)}
			<option value={option.value}>{option.label}</option>
		{/each}

		{#each groups as group (group.heading)}
			<optgroup label={group.heading}>
				{#each group.options as option (option.value)}
					<option value={option.value}>{option.label}</option>
				{/each}
			</optgroup>
		{/each}
	</select>
</div>
