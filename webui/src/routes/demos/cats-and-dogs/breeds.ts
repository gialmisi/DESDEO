/**
 * Breed group names for the cats and dogs demo.
 *
 * The cat and dog breed problems carry a `breed_id` decision variable whose
 * value indexes into these lists. The lists are generated from the survey data
 * shipped with DESDEO by `scripts/generate_breed_names.py`, so they must be
 * regenerated if the data files change.
 */

export const CAT_BREED_NAMES: string[] = [
	'Abyssinian',
	'American_Curl',
	'Bengal',
	'British',
	'Burmese',
	'Cornish_Rex',
	'European',
	'House_cat',
	'Korat',
	'Landrace_Cat_Longhair',
	'Landrace_Cat_Shorthair',
	'Maine_Coon',
	'Norwegian_Forest_Cat',
	'Ocicat',
	'Oriental',
	'Other',
	'Persian_and_Exotic',
	'Ragdoll',
	'Russian_Blue',
	'Sacred_Birman',
	'Siamese_and_Balinese',
	'Siberian_and_Neva_Masquerade',
	'Somali',
	'Sphynx_and_Devon_Rex',
	'Turkish_Angora',
	'Turkish_Van',
];

export const DOG_BREED_NAMES: string[] = [
	'Australian_Shepherd',
	'Belgian_shepherd_dogs',
	'Bernese_Mountain_Dogs',
	'Bichon_type_dogs',
	'Border_Collie',
	'Brachycephalic_dogs',
	'Bull_type_terriers',
	'Chinese_Crested_Dog',
	'Collie_Rough',
	'Collie_Smooth',
	'Dachshunds',
	'English_herders',
	'European_sighthounds',
	'Fighting_dogs',
	'Finnish_Lapphund',
	'German_Shepherd_Dog',
	'German_spitz_related',
	'Golden_Retriever',
	'Hunting_terriers',
	'Jack_Russell_Terrier',
	'Japanese_Asian_primitive',
	'Labrador_Retriever',
	'Lagotto_Romagnolo_Romagna_Water_Dog',
	'Lapponian_Herder',
	'Livestock_guardian_dogs',
	'Mastiff_type_dogs',
	'Middle_European_herders',
	'Middle_European_utility_dogs',
	'Miniature_Pinscher',
	'Miniature_Schnauzer',
	'Mixed_breed',
	'Northern_companion_spitz',
	'Northern_hunting_spitz',
	'Nova_Scotia_Duck_Tolling_Retriever',
	'Other_breed',
	'Other_companion_dogs',
	'Parson_type_terriers',
	'Pinschers_Schnauzers',
	'Pointers',
	'Poodles',
	'Primitive_sighthounds',
	'Retrievers_flushing_dogs',
	'Scenthounds',
	'Schapendoes',
	'Shetland_Sheepdog',
	'Sled_dogs',
	'Spanish_Water_Dog',
	'Teacup_dogs',
	'Welsh_Corgis',
	'Whippet',
	'White_Swiss_Shepherd_Dog',
	'Yard_terriers',
];

/** The animals the demo can find a breed for. */
export type Animal = 'cat' | 'dog';

/** Returns the breed group names of the given animal, ordered by breed id. */
export function breedNames(animal: Animal): string[] {
	return animal === 'cat' ? CAT_BREED_NAMES : DOG_BREED_NAMES;
}

/**
 * Returns the name of the breed group with the given id, or a placeholder if
 * the id is not one of the breed groups of the animal.
 */
export function breedName(animal: Animal, breedId: number | null | undefined): string {
	const names = breedNames(animal);
	if (breedId == null || breedId < 0 || breedId >= names.length) {
		return 'Unknown breed group';
	}
	return names[breedId];
}

/** Turns a breed group name into something readable, for example "Maine Coon". */
export function breedLabel(name: string): string {
	return name.replaceAll('_', ' ');
}

/**
 * Returns the path to the photograph of a breed group. The images live in
 * `static/animal_pics` and are fetched by `scripts/fetch_breed_images.py`. A
 * missing image is handled by the demo, which falls back to showing the name
 * on its own.
 */
export function breedImagePath(name: string): string {
	return `/animal_pics/${name}.jpg`;
}
