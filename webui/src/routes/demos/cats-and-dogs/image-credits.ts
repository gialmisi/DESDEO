/**
 * Credits for the breed photographs shown by the cats and dogs demo.
 *
 * The photographs come from Wikimedia Commons and nearly all of them are
 * licensed so that the author has to be credited. The credits are written next
 * to the images by `scripts/fetch_breed_images.py`, and the demo shows them
 * under the photograph of the chosen breed.
 */

import type { Animal } from './breeds';

export type BreedImageCredit = {
	animal: Animal;
	file: string;
	/** The breed whose photograph stands in for the breed group. */
	represented_by: string;
	commons_file: string;
	license: string;
	author: string;
	attribution_required: boolean;
	/** The description page of the file on Wikimedia Commons. */
	source: string;
};

/**
 * Loads the credits of the breed photographs. Returns an empty record if the
 * credits file is missing, so that a demo without images still runs.
 */
export async function loadBreedImageCredits(
	fetchImpl: typeof fetch = fetch
): Promise<Record<string, BreedImageCredit>> {
	try {
		const response = await fetchImpl('/animal_pics/credits.json');
		if (!response.ok) return {};
		return (await response.json()) as Record<string, BreedImageCredit>;
	} catch {
		return {};
	}
}
