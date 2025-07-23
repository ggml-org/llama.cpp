<script lang="ts">
	import { Input } from '$lib/components/ui/input';
	import { Textarea } from '$lib/components/ui/textarea';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import type { FieldConfig, ConfigValue } from '$lib/types/settings';

	interface Props {
		field: FieldConfig;
		value: ConfigValue;
		defaultValue: ConfigValue;
		onChange: (value: ConfigValue) => void;
	}

	let { field, value, defaultValue, onChange }: Props = $props();

	function handleInputChange(event: Event) {
		const target = event.currentTarget as HTMLInputElement | HTMLTextAreaElement;
		onChange(target.value);
	}

	function handleCheckboxChange(checked: boolean | 'indeterminate') {
		onChange(checked === true);
	}
</script>

<div class="space-y-2">
	{#if field.type === 'input'}
		<label for={field.key} class="block text-sm font-medium">
			{field.label}
		</label>
		<Input
			id={field.key}
			value={String(value || '')}
			onchange={handleInputChange}
			placeholder={`Default: ${defaultValue || 'none'}`}
			class="max-w-md"
		/>
		{#if field.help}
			<p class="text-muted-foreground mt-1 text-xs">
				{field.help}
			</p>
		{/if}
	{:else if field.type === 'textarea'}
		<label for={field.key} class="block text-sm font-medium">
			{field.label}
		</label>
		<Textarea
			id={field.key}
			value={String(value || '')}
			onchange={handleInputChange}
			placeholder={`Default: ${defaultValue || 'none'}`}
			class="min-h-[100px] max-w-2xl"
		/>
		{#if field.help}
			<p class="text-muted-foreground mt-1 text-xs">
				{field.help}
			</p>
		{/if}
	{:else if field.type === 'checkbox'}
		<div class="flex items-start space-x-3">
			<Checkbox
				id={field.key}
				checked={Boolean(value)}
				onCheckedChange={handleCheckboxChange}
				class="mt-1"
			/>
			<div class="space-y-1">
				<label
					for={field.key}
					class="text-sm font-medium leading-none cursor-pointer"
				>
					{field.label}
				</label>
				{#if field.help}
					<p class="text-muted-foreground text-xs">
						{field.help}
					</p>
				{/if}
			</div>
		</div>
	{/if}
</div>
