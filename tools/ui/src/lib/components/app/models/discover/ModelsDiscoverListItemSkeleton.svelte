<script lang="ts">
	import { Skeleton } from '$lib/components/ui/skeleton';
</script>

<!-- Static skeleton of ModelsDiscoverListItem: avatar, name and badge rows. -->
<li>
	<div class="flex w-full items-start gap-2.5 rounded-lg p-2.5 text-left">
		<Skeleton class="h-9 w-9 rounded-md" />

		<div class="min-w-0 flex-1 space-y-1.5">
			<Skeleton class="h-4 w-48 max-w-full" />

			<div class="flex items-center gap-1">
				<Skeleton class="h-3.5 w-12 rounded" />

				<Skeleton class="h-3.5 w-14 rounded" />

				<Skeleton class="h-3.5 w-10 rounded" />
			</div>
		</div>
	</div>
</li>
