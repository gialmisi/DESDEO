import type {
	NautilusNavigatorInitRequest,
	NautilusNavigatorRecomputeRequest,
	NautilusNavigatorSegmentResponse,
	ProblemGetRequest,
	ProblemInfo
} from '$lib/gen/models';
import type {
	getProblemProblemGetPostResponse,
	initializeMethodNautilusNavigatorInitializePostResponse,
	recomputeMethodNautilusNavigatorRecomputePostResponse
} from '$lib/gen/endpoints/DESDEOFastAPI';
import {
	getProblemProblemGetPost,
	initializeMethodNautilusNavigatorInitializePost,
	recomputeMethodNautilusNavigatorRecomputePost
} from '$lib/gen/endpoints/DESDEOFastAPI';

export async function initialize_navigator(
	request: NautilusNavigatorInitRequest
): Promise<NautilusNavigatorSegmentResponse | null> {
	const response: initializeMethodNautilusNavigatorInitializePostResponse =
		await initializeMethodNautilusNavigatorInitializePost(request);

	if (response.status !== 200) {
		console.error('NAUTILUS Navigator initialize failed.', response.status);
		return null;
	}

	return response.data;
}

export async function recompute_navigator(
	request: NautilusNavigatorRecomputeRequest
): Promise<NautilusNavigatorSegmentResponse | null> {
	const response: recomputeMethodNautilusNavigatorRecomputePostResponse =
		await recomputeMethodNautilusNavigatorRecomputePost(request);

	if (response.status !== 200) {
		console.error('NAUTILUS Navigator recompute failed.', response.status);
		return null;
	}

	return response.data;
}

export async function fetch_problem_info(request: ProblemGetRequest): Promise<ProblemInfo | null> {
	const response: getProblemProblemGetPostResponse = await getProblemProblemGetPost(request);

	if (response.status !== 200) {
		console.log('Could not fetch problem info.', response.status);
		return null;
	}

	return response.data;
}
