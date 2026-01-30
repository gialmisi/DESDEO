import type { ProblemInfo } from '$lib/gen/models';
import { getProblemsProblemAllGet } from '$lib/gen/endpoints/DESDEOFastAPI';
import { handleAuthFailure } from '$lib/handlers/auth';

type HandlerResult<T> = { ok: true; data: T } | { ok: false; error: string };

export async function fetchProblems(
	fetchImpl?: typeof fetch
): Promise<HandlerResult<ProblemInfo[]>> {
	const response = await getProblemsProblemAllGet({ fetchImpl });

	if (response.status !== 200) {
		handleAuthFailure(response.status);
		return { ok: false, error: 'Failed to fetch problems.' };
	}

	return { ok: true, data: response.data };
}
