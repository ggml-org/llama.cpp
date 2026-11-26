// The bundler's flags, read when present: Vite sets import.meta.env.DEV in development.
interface ImportMeta {
	readonly env?: { readonly DEV?: boolean };
}
