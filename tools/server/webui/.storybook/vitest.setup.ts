import * as a11yAddonAnnotations from '@storybook/addon-a11y/preview';
import { setProjectAnnotations } from '@storybook/sveltekit';
import * as previewAnnotations from './preview';
import { beforeAll } from 'vitest';

const project = setProjectAnnotations([a11yAddonAnnotations, previewAnnotations]);

beforeAll(async () => {
	if (project.beforeAll) {
		await project.beforeAll();
	}
});
