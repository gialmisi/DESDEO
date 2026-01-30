import type { PageLoad } from './$types';
import { fetchGroupPageData } from './handler';

export const load: PageLoad = async ({ fetch }) => {
	const res = await fetchGroupPageData(fetch);
	if (!res.ok) {
		throw new Error(res.error);
	}

	return {
		problemList: res.data.problemList,
		groupList: res.data.groupList
	};
};
