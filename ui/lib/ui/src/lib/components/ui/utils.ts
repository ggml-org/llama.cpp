import { type ClassValue, clsx } from 'clsx';
import { extendTailwindMerge } from 'tailwind-merge';
import { createTV } from 'tailwind-variants';

// the type, corner, icon and height roles from src/lib/styles/tokens/semantic.css, so text-label is
// merged as a font size and rounded-control as a radius rather than dropped or kept twice
const mergeConfig = {
	extend: {
		classGroups: {
			h: [{ h: ['control-sm', 'control-md', 'control-lg'] }],
			size: [{ size: ['icon-xs', 'icon-sm', 'icon-md', 'icon-lg'] }]
		},
		theme: {
			radius: ['control', 'panel', 'card', 'container', 'surface', 'surface-lg'],
			text: ['body', 'body-sm', 'label', 'label-sm', 'caption', 'title', 'heading']
		}
	}
};
const twMerge = extendTailwindMerge(mergeConfig);

/** tailwind-variants with the same merge rules as cn, so variants keep the token roles. */
export const tv = createTV({ twMergeConfig: mergeConfig });

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type WithoutChild<T> = T extends { child?: any } ? Omit<T, 'child'> : T;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type WithoutChildren<T> = T extends { children?: any } ? Omit<T, 'children'> : T;
export type WithoutChildrenOrChild<T> = WithoutChildren<WithoutChild<T>>;
export type WithElementRef<T, U extends HTMLElement = HTMLElement> = T & { ref?: U | null };
