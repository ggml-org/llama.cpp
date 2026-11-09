<script lang="ts">
	import { colorLevelTextClass } from './context-gauge';
	import { CONTEXT_GAUGE_DIAL_SIZES } from '$lib/constants';
	import { ColorLevel } from '$lib/enums';

	interface Props {
		percent: number | null;
		level: ColorLevel;
		/** sm inline, md a phone's inline touch target, key the icon of a phone's corner key */
		size?: keyof typeof CONTEXT_GAUGE_DIAL_SIZES;
	}

	let { level, percent, size = 'sm' }: Props = $props();

	const dial = $derived(CONTEXT_GAUGE_DIAL_SIZES[size]);
	const center = $derived(dial.grid / 2);
	const circumference = $derived(2 * Math.PI * dial.radius);

	// on the key, at rest, the arc takes the icon colour of the key, not the muted one
	const strokeLevelClass = $derived(
		size === 'key' && level !== ColorLevel.WARNING && level !== ColorLevel.CRITICAL
			? 'text-accent-foreground'
			: colorLevelTextClass(level)
	);
</script>

<svg class={dial.sizeClass} fill="none" viewBox="0 0 {dial.grid} {dial.grid}">
	<circle
		cx={center}
		cy={center}
		r={dial.radius}
		stroke="currentColor"
		stroke-opacity={dial.trackOpacity}
		stroke-width={dial.strokeWidth}
	/>

	<circle
		class="transition-colors duration-300 {strokeLevelClass}"
		cx={center}
		cy={center}
		r={dial.radius}
		stroke="currentColor"
		stroke-dasharray={circumference}
		stroke-dashoffset={percent !== null ? circumference * (1 - percent / 100) : circumference}
		stroke-linecap="round"
		stroke-width={dial.strokeWidth}
		transform="rotate(-90 {center} {center})"
	/>
</svg>
