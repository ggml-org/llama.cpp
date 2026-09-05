/**
 * A web-only build is a static build for a web host, with no llama-server
 * behind it. Every request must go through the WebRTC tunnel, so the user has
 * to paste the access code of a llama-connect instance before the app can run.
 *
 * Enabled by setting LLAMA_UI_WEB_ONLY when building.
 */
export const IS_WEB_ONLY = __WEB_ONLY__;
