#!/usr/bin/env node
// Decides whether this library is a version npm does not have yet, for
// .github/workflows/ui-lib-publish.yml. It reads package.json next to its own folder, so it
// runs from anywhere.
//
// The version in package.json is the release: this takes it, asks the registry what it holds,
// and says whether to publish, under which dist-tag, and why not when not. The version already
// being on npm is the case that let a release drift once (the library moved, the version did
// not), so it is reported as a verdict rather than passed over in silence.
//
// It writes its findings to $GITHUB_OUTPUT when that is set, and always exits 0 unless the
// registry could not be read: the workflow decides between warning and failing.

import { readFileSync, appendFileSync } from 'node:fs';

const REGISTRY = process.env.NPM_REGISTRY_URL ?? 'https://registry.npmjs.org';

const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
const { name, version } = pkg;

if (!name || !version) {
	console.error(`package.json holds no name or version (got ${name}@${version})`);
	process.exit(1);
}

// @scoped/name goes in the registry path with only the slash escaped
const url = `${REGISTRY}/${name.replace('/', '%2F')}`;
let doc;
try {
	const res = await fetch(url);
	if (res.status === 404) {
		doc = null; // never published
	} else if (!res.ok) {
		console.error(`${url} answered ${res.status}`);
		process.exit(1);
	} else {
		doc = await res.json();
	}
} catch (err) {
	console.error(`${url} could not be read: ${err.message}`);
	process.exit(1);
}

const latest = doc?.['dist-tags']?.latest ?? null;
const publishedHere = Boolean(doc?.versions?.[version]);
// 1.2.3-beta.4 is a prerelease: it goes under next, so latest keeps pointing at a stable build
const prerelease = /[-+]/.test(version);

let verdict = 'new';
let shouldPublish = true;
let note = `${name}@${version} is not on npm${latest ? `, which holds ${latest}` : ''}`;

if (publishedHere) {
	verdict = 'published';
	shouldPublish = false;
	note = `${name}@${version} is already on npm. Bump the version in ui/lib/ui/package.json to release these changes.`;
} else if (latest && isLower(version, latest)) {
	verdict = 'stale';
	shouldPublish = false;
	note = `${name}@${version} is older than the ${latest} on npm.`;
}

const tag = prerelease ? 'next' : 'latest';

console.log(`package.json : ${name}@${version}`);
console.log(`npm latest   : ${latest ?? 'nothing published yet'}`);
console.log(`that version : ${publishedHere ? 'already on npm' : 'not on npm'}`);
console.log(`verdict      : ${verdict} (publish: ${shouldPublish}, dist-tag: ${tag})`);
console.log(`               ${note}`);

if (process.env.GITHUB_OUTPUT) {
	appendFileSync(
		process.env.GITHUB_OUTPUT,
		[
			`name=${name}`,
			`version=${version}`,
			`latest=${latest ?? ''}`,
			`verdict=${verdict}`,
			`should_publish=${shouldPublish}`,
			`tag=${tag}`,
			`note=${note}`,
			'',
		].join('\n'),
	);
}

/** Compare two semver versions, ignoring any prerelease suffix, so 1.2.3-rc.1 sits on 1.2.3. */
function isLower(a, b) {
	const [x, y] = [core(a), core(b)];
	for (let i = 0; i < 3; i++) if (x[i] !== y[i]) return x[i] < y[i];
	return false;
}

function core(v) {
	return v.replace(/[-+].*$/, '').split('.').map((n) => Number(n) || 0);
}
