import type { GroupPublic, ProblemInfo } from '$lib/gen/models';
import {
	getGroupInfoGdmGetGroupInfoPost,
	getProblemProblemGetPost
} from '$lib/gen/endpoints/DESDEOFastAPI';
import { handleAuthFailure } from '$lib/handlers/auth';

type HandlerResult<T> = { ok: true; data: T } | { ok: false; error: string };

export async function fetchGroupAndProblem(
	groupId: number,
	fetchImpl?: typeof fetch
): Promise<HandlerResult<{ group: GroupPublic; problem: ProblemInfo }>> {
	type FetchOptions = RequestInit & { fetchImpl?: typeof fetch };
	const requestOptions = fetchImpl ? ({ fetchImpl } as FetchOptions) : undefined;
	const groupResponse = await getGroupInfoGdmGetGroupInfoPost({ group_id: groupId }, requestOptions);

	if (groupResponse.status !== 200) {
		handleAuthFailure(groupResponse.status);
		return { ok: false, error: 'Failed to fetch group info.' };
	}

	const problemResponse = await getProblemProblemGetPost(
		{ problem_id: groupResponse.data.problem_id },
		requestOptions
	);

	if (problemResponse.status !== 200) {
		handleAuthFailure(problemResponse.status);
		return { ok: false, error: 'Failed to fetch problem info.' };
	}

	return { ok: true, data: { group: groupResponse.data, problem: problemResponse.data } };
}
