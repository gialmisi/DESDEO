import type {
	NautilusNavigatorInitializeRequest,
	NautilusNavigatorInitializeResponse,
	NautilusNavigatorNavigateRequest,
	NautilusNavigatorNavigateResponse,
	ProblemGetRequest,
	ProblemInfo
} from '$lib/gen/models';
import {
	getProblemProblemGetPost,
	initializeNavigatorMethodNautilusNavigatorInitializePost,
	navigateNavigatorMethodNautilusNavigatorNavigatePost,
	type getProblemProblemGetPostResponse,
	type initializeNavigatorMethodNautilusNavigatorInitializePostResponse,
	type navigateNavigatorMethodNautilusNavigatorNavigatePostResponse
} from '$lib/gen/endpoints/DESDEOFastAPI';

export async function fetch_problem_info(request: ProblemGetRequest): Promise<ProblemInfo | null> {
	const response: getProblemProblemGetPostResponse = await getProblemProblemGetPost(request);

	if (response.status !== 200) {
		console.error('Could not fetch problem info.', response.status);
		return null;
	}

	return response.data;
}

export async function initialize_navigator(
	request: NautilusNavigatorInitializeRequest
): Promise<NautilusNavigatorInitializeResponse | null> {
	const response: initializeNavigatorMethodNautilusNavigatorInitializePostResponse =
		await initializeNavigatorMethodNautilusNavigatorInitializePost(request);

	if (response.status !== 200) {
		console.error('Failed to initialize NAUTILUS Navigator.', response.status);
		return null;
	}

	return response.data;
}

export async function navigate_navigator(
	request: NautilusNavigatorNavigateRequest
): Promise<NautilusNavigatorNavigateResponse | null> {
	const response: navigateNavigatorMethodNautilusNavigatorNavigatePostResponse =
		await navigateNavigatorMethodNautilusNavigatorNavigatePost(request);

	if (response.status !== 200) {
		console.error('Failed to navigate NAUTILUS Navigator.', response.status);
		return null;
	}

	return response.data;
}
