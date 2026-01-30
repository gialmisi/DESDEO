import type { ProblemInfo } from '$lib/gen/models';
import { getProblemsInfoProblemAllInfoGet } from '$lib/gen/endpoints/DESDEOFastAPI';
import { handleAuthFailure } from '$lib/handlers/auth';

type HandlerResult<T> = { ok: true; data: T } | { ok: false; error: string };

export async function fetchProblemInfoList(
	fetchImpl?: typeof fetch
): Promise<HandlerResult<ProblemInfo[]>> {
	const response = await getProblemsInfoProblemAllInfoGet({ fetchImpl });

	if (response.status !== 200) {
		handleAuthFailure(response.status);
		return { ok: false, error: 'Failed to fetch problems.' };
	}

	return { ok: true, data: response.data };
}
