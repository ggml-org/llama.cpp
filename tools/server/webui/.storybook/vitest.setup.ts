import { setProjectAnnotations } from '@storybook/sveltekit';
import * as previewAnnotations from './preview';
import { beforeAll } from 'vitest';

const project = setProjectAnnotations([previewAnnotations]);

beforeAll(async () => {
	if (project.beforeAll) {
		await project.beforeAll();
	}
});
