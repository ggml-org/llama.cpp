const logs = [];
const fmt = (value) => {
	if (typeof value === 'string') return value;
	try {
		return JSON.stringify(value);
	} catch {
		return String(value);
	}
};
const capture =
	(level, prefix) =>
	(...args) => {
		logs.push(prefix + args.map(fmt).join(' '));
	};
console.log = capture('log', '');
console.info = capture('info', '');
console.debug = capture('debug', '');
console.warn = capture('warn', 'warn: ');
console.error = capture('error', 'error: ');
self.onmessage = async (event) => {
	const reply = { logs, result: null, error: null };
	try {
		const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
		// Inject `nerdamer` (from preloaded nerdamer-prime.js) into the execution context.
		// User code can use: nerdamer(expr), nerdamer.solve(), nerdamer.derivative(),
		// nerdamer.integrate(), nerdamer.expand(), nerdamer.factor(), nerdamer.simplify(),
		// nerdamer.laplace(), nerdamer.limit(), nerdamer.series(), etc.
		const value = await new AsyncFunction('nerdamer', event.data.code)(self.nerdamer);
		if (value !== undefined) reply.result = fmt(value);
	} catch (err) {
		reply.error = err instanceof Error ? err.stack || err.message : String(err);
	}
	self.postMessage(reply);
};
