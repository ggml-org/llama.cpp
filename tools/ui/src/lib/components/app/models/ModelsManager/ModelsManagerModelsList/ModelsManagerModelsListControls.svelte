<script lang="ts">
	import type { ModelsListFilter } from '../utils';
	import { Heart } from '@lucide/svelte';
	import { Input } from '$lib/components/ui/input';

	interface Props {
		filter?: string;
		view?: ModelsListFilter;
	}

	let { filter = $bindable(''), view = $bindable<ModelsListFilter>('all') }: Props = $props();

	const chipClass =
		'cursor-pointer rounded-full px-2.5 py-1 text-xs font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-ring';
</script>

<div class="shrink-0 space-y-2">
	<Input bind:value={filter} class="h-9 text-sm" placeholder="Filter models..." />

	<div class="flex flex-wrap items-center gap-1.5">
		<button
			aria-pressed={view === 'all'}
			class={[
				chipClass,
				view === 'all'
					? 'bg-primary text-primary-foreground'
					: 'border border-border text-muted-foreground hover:bg-muted'
			]}
			onclick={() => (view = 'all')}
			type="button"
		>
			All
		</button>

		<button
			aria-pressed={view === 'favorites'}
			class={[
				chipClass,
				'inline-flex items-center gap-1',
				view === 'favorites'
					? 'bg-primary text-primary-foreground'
					: 'border border-border text-muted-foreground hover:bg-muted'
			]}
			onclick={() => (view = 'favorites')}
			type="button"
		>
			<Heart class="h-3 w-3" />

			favorites
		</button>

		<button
			aria-pressed={view === 'loaded'}
			class={[
				chipClass,
				view === 'loaded'
					? 'bg-primary text-primary-foreground'
					: 'border border-border text-muted-foreground hover:bg-muted'
			]}
			onclick={() => (view = 'loaded')}
			type="button"
		>
			loaded
		</button>
	</div>
</div>
