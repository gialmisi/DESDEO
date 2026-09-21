"""Generate the breed group name tables used by the cats and dogs demo.

The cat and dog breed problems carry a `breed_id` decision variable, which is
an index into the sorted list of breed group names of the animal. The web UI
needs those names to tell the decision maker which breed a solution belongs
to, so they are written out as a TypeScript module here rather than being
typed in by hand.

Run from the repository root:
    python scripts/generate_breed_names.py
"""
# ruff: noqa: T201

from pathlib import Path

from desdeo.problem.testproblems import cat_breed_group_names, dog_breed_group_names

OUTPUT_PATH = Path(__file__).parents[1] / "webui" / "src" / "routes" / "demos" / "cats-and-dogs" / "breeds.ts"

TEMPLATE = """/**
 * Breed group names for the cats and dogs demo.
 *
 * The cat and dog breed problems carry a `breed_id` decision variable whose
 * value indexes into these lists. The lists are generated from the survey data
 * shipped with DESDEO by `scripts/generate_breed_names.py`, so they must be
 * regenerated if the data files change.
 */

export const CAT_BREED_NAMES: string[] = [
{cat_names}
];

export const DOG_BREED_NAMES: string[] = [
{dog_names}
];

/** The animals the demo can find a breed for. */
export type Animal = 'cat' | 'dog';

/** Returns the breed group names of the given animal, ordered by breed id. */
export function breedNames(animal: Animal): string[] {{
\treturn animal === 'cat' ? CAT_BREED_NAMES : DOG_BREED_NAMES;
}}

/**
 * Returns the name of the breed group with the given id, or a placeholder if
 * the id is not one of the breed groups of the animal.
 */
export function breedName(animal: Animal, breedId: number | null | undefined): string {{
\tconst names = breedNames(animal);
\tif (breedId == null || breedId < 0 || breedId >= names.length) {{
\t\treturn 'Unknown breed group';
\t}}
\treturn names[breedId];
}}

/** Turns a breed group name into something readable, for example "Maine Coon". */
export function breedLabel(name: string): string {{
\treturn name.replaceAll('_', ' ');
}}

/**
 * Returns the path to the photograph of a breed group. The images live in
 * `static/animal_pics` and are fetched by `scripts/fetch_breed_images.py`. A
 * missing image is handled by the demo, which falls back to showing the name
 * on its own.
 */
export function breedImagePath(name: string): string {{
\treturn `/animal_pics/${{name}}.jpg`;
}}
"""


def as_typescript_array(names: list[str]) -> str:
    """Format a list of names as the body of a TypeScript array literal.

    Args:
        names (list[str]): the names to format.

    Returns:
        str: the formatted array body, one name per line.
    """
    return "\n".join(f"\t'{name}'," for name in names)


if __name__ == "__main__":
    OUTPUT_PATH.write_text(
        TEMPLATE.format(
            cat_names=as_typescript_array(cat_breed_group_names()),
            dog_names=as_typescript_array(dog_breed_group_names()),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH}")
    print(f"{len(cat_breed_group_names())} cat breed groups, {len(dog_breed_group_names())} dog breed groups")
