/**
 * Records Llama Chat's demo videos with the harness in ui/lib/demo:
 *
 *   node demo/cli.ts pr <number>   record the ```ts demo block from a PR description, a video per test
 *                                  (--trust <sha> to record someone else's PR at that head commit)
 *   node demo/cli.ts docs          record every docs/*.demo.ts script in DEMO_STORAGE
 *   node demo/cli.ts traces        write the built-in traces, and push them to DEMO_TRACES_DATASET
 *
 * Settings come from ui/apps/chat/.env (see .env.example).
 */

import { DEMO_TRACES, generateTrace } from './fixtures/traces.ts';
import { runDemoCli } from '@ggml-org/llama-demo/cli';
import { execFileSync } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const DEMO_DIR = import.meta.dirname;
const TRACES_DIR = join(DEMO_DIR, 'traces');

/** Writes the built-in traces to demo/traces/ and uploads them to the traces dataset. */
function writeTraces(): void {
	rmSync(TRACES_DIR, { force: true, recursive: true });
	mkdirSync(TRACES_DIR, { recursive: true });

	for (const name of Object.keys(DEMO_TRACES)) {
		writeFileSync(join(TRACES_DIR, `${name}.jsonl`), `${generateTrace(name)}\n`);
		console.log(`wrote demo/traces/${name}.jsonl`);
	}

	const dataset = process.env.DEMO_TRACES_DATASET;

	if (!dataset) {
		console.log('DEMO_TRACES_DATASET is not set: not uploading.');

		return;
	}

	execFileSync(
		'hf',
		[
			'upload',
			dataset,
			TRACES_DIR,
			'.',
			'--repo-type',
			'dataset',
			'--commit-message',
			'Update demo traces'
		],
		{ stdio: 'inherit' }
	);
}

await runDemoCli({
	commands: { traces: writeTraces },
	demoDir: DEMO_DIR,
	dir: resolve(DEMO_DIR, '..'),
	// Chat builds Llama UI from source, and DEMO_STORYBOOK=ui runs its Storybook
	linkNodeModules: ['../../lib/ui']
});
