/** How a sign-in that left for the authorization page comes back. */
export enum HfOAuthFlowMode {
	/** In a popup, which can close once the tokens are in. */
	POPUP = 'popup',
	/** In the tab that started it, which goes back to where it was. */
	REDIRECT = 'redirect'
}

export enum HfOAuthGrantType {
	AUTHORIZATION_CODE = 'authorization_code',
	REFRESH_TOKEN = 'refresh_token'
}

/** Names of the OAuth request and callback parameters. */
export enum HfOAuthParam {
	CLIENT_ID = 'client_id',
	CODE = 'code',
	CODE_CHALLENGE = 'code_challenge',
	CODE_CHALLENGE_METHOD = 'code_challenge_method',
	CODE_VERIFIER = 'code_verifier',
	ERROR = 'error',
	ERROR_DESCRIPTION = 'error_description',
	GRANT_TYPE = 'grant_type',
	REDIRECT_URI = 'redirect_uri',
	REFRESH_TOKEN = 'refresh_token',
	RESPONSE_TYPE = 'response_type',
	SCOPE = 'scope',
	STATE = 'state'
}

/** Error codes the authorization page sends back that get their own wording. */
export enum HfOAuthErrorCode {
	ACCESS_DENIED = 'access_denied'
}

/** What an access token may do, as `/api/whoami-v2` reports it. */
export enum HfTokenRole {
	FINE_GRAINED = 'fineGrained',
	READ = 'read',
	WRITE = 'write'
}
