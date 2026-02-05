import type { PageLoad } from './$types';
import { fetchProblemInfoList } from './handler';

export const load: PageLoad = async ({ fetch }) => {
	const res = await fetchProblemInfoList(fetch);
	if (!res.ok) throw new Error(res.error);

	return {
		problemList: res.data
	};
};
