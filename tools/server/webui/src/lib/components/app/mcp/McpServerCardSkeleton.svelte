<script lang="ts">
	import * as Card from '$lib/components/ui/card';
	import { Skeleton } from '$lib/components/ui/skeleton';
</script>

<Card.Root class="grid gap-3 p-4">
	<div class="flex items-center justify-between gap-4">
		<div class="flex items-center gap-2">
			<Skeleton class="h-5 w-5 rounded" />
			<Skeleton class="h-5 w-28" />
			<Skeleton class="h-5 w-12 rounded-full" />
		</div>
		<Skeleton class="h-6 w-11 rounded-full" />
	</div>

	<div class="flex flex-wrap gap-1.5">
		<Skeleton class="h-5 w-14 rounded-full" />
		<Skeleton class="h-5 w-12 rounded-full" />
		<Skeleton class="h-5 w-16 rounded-full" />
	</div>

	<div class="space-y-1.5">
		<Skeleton class="h-4 w-40" />
		<Skeleton class="h-4 w-52" />
	</div>

	<Skeleton class="h-3.5 w-36" />

	<div class="flex justify-end gap-2">
		<Skeleton class="h-8 w-8 rounded" />
		<Skeleton class="h-8 w-8 rounded" />
		<Skeleton class="h-8 w-8 rounded" />
	</div>
</Card.Root>
