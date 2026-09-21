/** Types specific to the cats and dogs demo. */

import type { ProblemInfo, SolverResults } from '$lib/gen/endpoints/DESDEOFastAPI';
import type { Animal } from './breeds';

/** One breed the reference point method suggested to the decision maker. */
export type Candidate = {
	/** The objective values of the candidate, keyed by objective symbol. */
	objectiveValues: Record<string, number>;
	/** The row of the survey data the candidate comes from. */
	rowIndex: number;
	/** The breed group of the candidate, as an index into the animal's breed names. */
	breedId: number;
	/** The name of the breed group, for example "Maine_Coon". */
	breedName: string;
};

/** Everything the demo knows about the animal the decision maker picked. */
export type DemoProblem = {
	animal: Animal;
	problem: ProblemInfo;
};

export type { ProblemInfo, SolverResults, Animal };
