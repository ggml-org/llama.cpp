<script lang="ts">
	import type { ColorLevel } from './context-gauge';
	import { colorLevelTextClass } from './context-gauge';

	interface Props {
		percent: number | null;
		level: ColorLevel;
		size?: 'sm' | 'md';
	}

	let { percent, level, size = 'sm' }: Props = $props();

	const RADIUS = 11;
	const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

	const strokeLevelClass = $derived(colorLevelTextClass(level));
	const dimensions = $derived(size === 'md' ? 'h-6 w-6' : 'h-5 w-5');
	const strokeWidth = $derived(size === 'md' ? 4 : 3);
</script>

<svg viewBox="0 0 32 32" fill="none" class={dimensions}>
	<circle
		cx="16"
		cy="16"
		r={RADIUS}
		stroke="currentColor"
		stroke-opacity="0.1"
		stroke-width={strokeWidth}
	/>

	<circle
		cx="16"
		cy="16"
		r={RADIUS}
		class="transition-colors duration-300 {strokeLevelClass}"
		stroke="currentColor"
		stroke-width={strokeWidth}
		stroke-linecap="round"
		stroke-dasharray={CIRCUMFERENCE}
		stroke-dashoffset={percent !== null ? CIRCUMFERENCE * (1 - percent / 100) : CIRCUMFERENCE}
		transform="rotate(-90 16 16)"
	/>
</svg>
