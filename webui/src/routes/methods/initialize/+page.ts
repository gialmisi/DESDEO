import type { PageLoad } from './$types';
import { fetchProblems } from './handler';

export const load: PageLoad = async ({ url, fetch }) => {
	const res = await fetchProblems(fetch);
	if (!res.ok) throw new Error(res.error);
	
	// Check if we have a group ID parameter, for selecting a method for GDM
	const groupId = url.searchParams.get('group');
	if (groupId) {
		return {
			problems: res.data,
			groupId: groupId
		};
	}

	// Regular single-user mode
	return { problems: res.data };
};
