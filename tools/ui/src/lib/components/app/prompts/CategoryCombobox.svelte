<script lang="ts">
	import { tick } from 'svelte';
	import { Check, ChevronsUpDown, Plus } from '@lucide/svelte';
	import * as Command from '$lib/components/ui/command';
	import * as Popover from '$lib/components/ui/popover';
	import { Button } from '$lib/components/ui/button';
	import { cn } from '$lib/components/ui/utils';

	interface Props {
		value?: string;
		categories: string[];
		class?: string;
		id?: string;
	}

	let { value = $bindable(''), categories = [], class: className, id }: Props = $props();

	let open = $state(false);
	let query = $state('');
	let triggerRef = $state<HTMLButtonElement>(null!);

	const trimmedQuery = $derived(query.trim());
	const isOther = $derived(!value.trim());
	const otherMatches = $derived(
		trimmedQuery === '' || 'other'.includes(trimmedQuery.toLowerCase())
	);
	const reserved = $derived(trimmedQuery.toLowerCase() === 'other');
	const isNewCategory = $derived(
		trimmedQuery.length > 0 &&
			!reserved &&
			!categories.some((c) => c.toLowerCase() === trimmedQuery.toLowerCase())
	);
	const filtered = $derived(
		categories.filter((c) => c.toLowerCase().includes(trimmedQuery.toLowerCase()))
	);

	function select(category: string) {
		value = category;
		closeAndFocusTrigger();
	}

	function clearCategory() {
		value = '';
		closeAndFocusTrigger();
	}

	function closeAndFocusTrigger() {
		open = false;
		query = '';
		tick().then(() => {
			triggerRef.focus();
		});
	}

	function handleOpenChange(next: boolean) {
		open = next;
		if (!next) query = '';
	}
</script>

<Popover.Root bind:open onOpenChange={handleOpenChange}>
	<Popover.Trigger bind:ref={triggerRef}>
		{#snippet child({ props })}
			<Button
				{...props}
				{id}
				variant="outline"
				role="combobox"
				aria-expanded={open}
				class={cn(
					'w-full justify-between font-normal',
					!value && 'text-muted-foreground',
					className
				)}
			>
				<span class="truncate">{value || 'Other'}</span>
				<ChevronsUpDown class="ms-2 size-4 shrink-0 opacity-50" />
			</Button>
		{/snippet}
	</Popover.Trigger>
	<Popover.Content class="min-w-[var(--bits-popover-anchor-width)] p-0">
		<Command.Root shouldFilter={false}>
			<Command.Input bind:value={query} placeholder="Search or type a new category..." />
			<Command.List>
				<Command.Empty>No category found.</Command.Empty>
				<Command.Group>
					{#if otherMatches}
						<Command.Item value="Other" onSelect={() => clearCategory()}>
							<Check class={cn('size-4', !isOther && 'text-transparent')} />
							Other
						</Command.Item>
					{/if}
					{#if isNewCategory}
						<Command.Item value={trimmedQuery} onSelect={() => select(trimmedQuery)}>
							<Plus class="size-4" />
							<span>Create <strong>{trimmedQuery}</strong></span>
						</Command.Item>
					{/if}
					{#each filtered as category (category)}
						<Command.Item value={category} onSelect={() => select(category)}>
							<Check class={cn('size-4', value !== category && 'text-transparent')} />
							{category}
						</Command.Item>
					{/each}
				</Command.Group>
			</Command.List>
		</Command.Root>
	</Popover.Content>
</Popover.Root>
