import type { PageLoad } from './$types';
import { getProblemsInfoProblemAllInfoGet } from '$lib/gen/endpoints/DESDEOFastAPI';
import type { ProblemInfo } from '$lib/gen/endpoints/DESDEOFastAPI';

/**
 * The demo is opened by its address rather than reached by clicking through the
 * app, so its data is loaded in the browser.
 *
 * Loading it while rendering on the server does not work as things stand. The
 * session cookies do reach the API once the fetch of the load event is used,
 * but SvelteKit then refuses to let the generated API client read the
 * `content-type` header of the response. Allowing that needs
 * `filterSerializedResponseHeaders` in a `handle` hook, which would change how
 * every route in the web UI is served, so it is left alone here.
 */
export const ssr = false;

/**
 * The demo works on the two breed problems seeded by
 * `desdeo/api/db_init_catsanddogs.py`, which it finds by name.
 */
export const load: PageLoad = async ({ fetch }) => {
	// The fetch of the load event is passed on so that SvelteKit can track the
	// request instead of warning about it going through window.fetch.
	const res = await getProblemsInfoProblemAllInfoGet({ fetchImpl: fetch } as RequestInit);
	if (res.status !== 200) throw new Error('Failed to fetch problems');
	return { problems: res.data as ProblemInfo[] };
};
