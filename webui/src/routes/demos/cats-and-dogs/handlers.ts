/**
 * API handlers for the cats and dogs demo.
 *
 * The demo drives the reference point method through the Orval generated
 * endpoint function, the same way the interactive method pages do. The
 * reference point method only exposes a solve endpoint, so the demo keeps the
 * iteration history itself rather than reading it back from the server.
 */

import { solveSolutionsMethodRpmSolvePost } from '$lib/gen/endpoints/DESDEOFastAPI';
import type { RPMSolveRequest, SolverResults } from '$lib/gen/endpoints/DESDEOFastAPI';
import { errorMessage, isLoading } from '../../../stores/uiState';
import { breedName } from './breeds';
import type { Animal, Candidate, ProblemInfo } from './types';

/**
 * Turns one solver result into a candidate breed.
 *
 * The breed group of a solution is carried by the `breed_id` decision
 * variable of the breed problems, so no lookup against the problem data is
 * needed here.
 */
function toCandidate(result: SolverResults, animal: Animal): Candidate {
	const variables = (result.optimal_variables ?? {}) as Record<string, number>;
	const breedId = Number(variables.breed_id);

	return {
		objectiveValues: (result.optimal_objectives ?? {}) as Record<string, number>,
		rowIndex: Number(variables.index),
		breedId,
		breedName: breedName(animal, breedId)
	};
}

/**
 * Runs one iteration of the reference point method.
 *
 * The method returns one solution for the given reference point and one more
 * for each objective, found by perturbing the reference point. On a problem
 * with a discrete representation several of those perturbations can land on
 * the same row of the data, so the duplicates are dropped before the
 * candidates are shown to the decision maker.
 *
 * @param problem The breed problem being solved.
 * @param animal The animal the problem is about, used to name the breeds.
 * @param referencePoint The reference point, ordered like `problem.objectives`.
 * @returns The distinct candidate breeds, or null if the iteration failed.
 */
export async function iterate(
	problem: ProblemInfo,
	animal: Animal,
	referencePoint: number[]
): Promise<Candidate[] | null> {
	isLoading.set(true);
	errorMessage.set(null);

	try {
		const aspirationLevels = problem.objectives.reduce(
			(levels, objective, index) => {
				levels[objective.symbol] = referencePoint[index];
				return levels;
			},
			{} as Record<string, number>
		);

		const request: RPMSolveRequest = {
			problem_id: problem.id,
			preference: {
				preference_type: 'reference_point',
				aspiration_levels: aspirationLevels
			}
		};

		const response = await solveSolutionsMethodRpmSolvePost(request);

		if (response.status !== 200) {
			errorMessage.set(`Could not find breeds for this reference point (status ${response.status}).`);
			return null;
		}

		const candidates = (response.data.solver_results ?? [])
			.filter((result) => result.success)
			.map((result) => toCandidate(result, animal));

		const seen = new Set<number>();
		return candidates.filter((candidate) => {
			if (seen.has(candidate.rowIndex)) {
				return false;
			}
			seen.add(candidate.rowIndex);
			return true;
		});
	} catch (error) {
		const message = error instanceof Error ? error.message : 'Unknown error';
		errorMessage.set(message);
		console.error('Error while iterating the reference point method:', message);
		return null;
	} finally {
		isLoading.set(false);
	}
}
