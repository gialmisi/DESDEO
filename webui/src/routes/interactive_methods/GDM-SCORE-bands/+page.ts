import type { PageLoad } from './$types';
import { fetchGroupAndProblem } from './handler';

type LoadData = {
	refreshToken?: string;
};

export const load: PageLoad<LoadData> = async ({ url, data, fetch }) => {
	const groupId = url.searchParams.get('group');
	if (!groupId) throw new Error('No group ID provided');

	const res = await fetchGroupAndProblem(parseInt(groupId, 10), fetch);
	if (!res.ok) throw new Error(res.error);
	return {
		problem: res.data.problem,
		refreshToken: data.refreshToken,
		group: res.data.group
	};
};
