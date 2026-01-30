import { get } from 'svelte/store';
import type { GroupPublic, ProblemInfo, UserPublic } from '$lib/gen/models';
import {
	getGroupInfoGdmGetGroupInfoPost,
	getProblemProblemGetPost
} from '$lib/gen/endpoints/DESDEOFastAPI';
import { appContext } from '../../stores/appContext';
import { bootstrapUser } from '$lib/bootstrap';
import { handleAuthFailure } from '$lib/handlers/auth';

type HandlerResult<T> = { ok: true; data: T } | { ok: false; error: string };

type GroupPageData = {
	groupList: GroupPublic[];
	problemList: ProblemInfo[];
};

export async function fetchGroupPageData(
	fetchImpl?: typeof fetch
): Promise<HandlerResult<GroupPageData>> {
	type FetchOptions = RequestInit & { fetchImpl?: typeof fetch };
	const requestOptions = fetchImpl ? ({ fetchImpl } as FetchOptions) : undefined;
	await bootstrapUser(appContext, fetchImpl);

	const user: UserPublic | null = get(appContext).user;
	if (!user) {
		return { ok: false, error: 'Not authenticated.' };
	}

	const groupIds = user.group_ids ?? [];
	if (groupIds.length === 0) {
		return { ok: true, data: { groupList: [], problemList: [] } };
	}

	const groupResponses = await Promise.all(
		groupIds.map((id) => getGroupInfoGdmGetGroupInfoPost({ group_id: id }, requestOptions))
	);

	const groupError = groupResponses.find((response) => response.status !== 200);
	if (groupError) {
		handleAuthFailure(groupError.status);
		return { ok: false, error: 'Failed to fetch group info.' };
	}

	const groupList = groupResponses.map((response) => response.data as GroupPublic);

	const problemResponses = await Promise.all(
		groupList.map((group) => getProblemProblemGetPost({ problem_id: group.problem_id }, requestOptions))
	);

	const problemError = problemResponses.find((response) => response.status !== 200);
	if (problemError) {
		handleAuthFailure(problemError.status);
		return { ok: false, error: 'Failed to fetch problem info.' };
	}

	const problemList = problemResponses.map((response) => response.data as ProblemInfo);

	return { ok: true, data: { groupList, problemList } };
}
