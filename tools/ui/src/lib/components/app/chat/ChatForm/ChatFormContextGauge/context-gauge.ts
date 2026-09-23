import { ColorLevel } from '$lib/enums';

const WARNING_THRESHOLD = 80;
const CRITICAL_THRESHOLD = 95;

export function colorLevelFromPercent(percent: number | null): ColorLevel {
	if (percent === null) return ColorLevel.NEUTRAL;

	if (percent >= CRITICAL_THRESHOLD) return ColorLevel.CRITICAL;

	if (percent >= WARNING_THRESHOLD) return ColorLevel.WARNING;

	return ColorLevel.OK;
}

export function colorLevelTextClass(level: ColorLevel): string {
	switch (level) {
		case ColorLevel.CRITICAL:
			return 'text-red-400';
		case ColorLevel.WARNING:
			return 'text-amber-400';
		case ColorLevel.OK:
			return 'text-muted-foreground';
		default:
			return 'text-muted-foreground';
	}
}

export function colorLevelBgClass(level: ColorLevel): string {
	switch (level) {
		case ColorLevel.CRITICAL:
			return 'bg-red-500';
		case ColorLevel.WARNING:
			return 'bg-amber-500';
		case ColorLevel.OK:
			return 'bg-green-500';
		default:
			return 'bg-muted';
	}
}
