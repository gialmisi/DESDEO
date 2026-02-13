import type {
	ScenarioRPMSolveRequest,
	ScenarioRPMState,
	ScenarioComparisonRequest,
	ScenarioComparisonResponse,
	ProblemGetRequest,
	ProblemInfo
} from '$lib/gen/models';

import type {
	solveMethodScenarioRpmSolvePostResponse,
	compareMethodScenarioRpmComparePostResponse,
	getStateMethodScenarioRpmGetStateStateIdGetResponse,
	getProblemProblemGetPostResponse
} from '$lib/gen/endpoints/DESDEOFastAPI';

import {
	solveMethodScenarioRpmSolvePost,
	compareMethodScenarioRpmComparePost,
	getStateMethodScenarioRpmGetStateStateIdGet,
	getProblemProblemGetPost
} from '$lib/gen/endpoints/DESDEOFastAPI';

export { fetch_sessions, create_session } from '../../methods/sessions/handler';

export async function fetch_problem_info(
	request: ProblemGetRequest
): Promise<ProblemInfo | null> {
	const response: getProblemProblemGetPostResponse =
		await getProblemProblemGetPost(request);
	if (response.status !== 200) {
		console.error('Could not fetch problem info.', response.status);
		return null;
	}
	return response.data;
}

export async function solve_scenario_rpm(
	request: ScenarioRPMSolveRequest
): Promise<ScenarioRPMState | null> {
	const response: solveMethodScenarioRpmSolvePostResponse =
		await solveMethodScenarioRpmSolvePost(request);
	if (response.status !== 200) {
		console.error('Scenario RPM solve failed.', response.status);
		return null;
	}
	return response.data;
}

export async function compare_solutions(
	request: ScenarioComparisonRequest
): Promise<ScenarioComparisonResponse | null> {
	const response: compareMethodScenarioRpmComparePostResponse =
		await compareMethodScenarioRpmComparePost(request);
	if (response.status !== 200) {
		console.error('Scenario comparison failed.', response.status);
		return null;
	}
	return response.data;
}

export async function fetch_scenario_rpm_state(
	stateId: number
): Promise<ScenarioRPMState | null> {
	const response: getStateMethodScenarioRpmGetStateStateIdGetResponse =
		await getStateMethodScenarioRpmGetStateStateIdGet(stateId);
	if (response.status !== 200) {
		console.error('Could not fetch scenario RPM state.', response.status);
		return null;
	}
	return response.data;
}
