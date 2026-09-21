<script lang="ts">
	/**
	 * +page.svelte (cats and dogs demo)
	 *
	 * @description
	 * A playful demonstration of interactive multiobjective optimization: the
	 * decision maker says what kind of cat or dog they would like, and the
	 * reference point method suggests breed groups that match.
	 *
	 * The problems behind the demo are built from two Finnish survey studies
	 * that charted the behaviour of cat and dog breeds. Each candidate is one
	 * non-dominated survey response, and its breed group is what gets shown to
	 * the decision maker. The problem formulation is not meant to be taken
	 * seriously, which the demo says out loud on its first screen.
	 *
	 * @stages
	 * 1. Choose: pick cats or dogs.
	 * 2. Explore: give a reference point, look at the suggested breeds, and
	 *    iterate as many times as wanted.
	 * 3. Result: the chosen breed, with a photograph.
	 *
	 * @notes
	 * - The demo drives `/method/rpm/solve` directly and keeps its own history,
	 *   because the reference point method has no save or finalize endpoints.
	 * - Breed group names come from `breeds.ts`, which is generated from the
	 *   survey data by `scripts/generate_breed_names.py`.
	 */
	import { onMount } from 'svelte';

	import { BaseLayout } from '$lib/components/custom/method_layout/index.js';
	import AppSidebar from '$lib/components/custom/preferences-bar/preferences-sidebar.svelte';
	import VisualizationsPanel from '$lib/components/custom/visualizations-panel/visualizations-panel.svelte';
	import LoadingSpinner from '$lib/components/custom/notifications/loading-spinner.svelte';
	import Alert from '$lib/components/custom/notifications/alert.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import * as Card from '$lib/components/ui/card/index.js';
	import * as Table from '$lib/components/ui/table/index.js';
	import { PREFERENCE_TYPES } from '$lib/constants';
	import { formatNumber, getDisplayAccuracy } from '$lib/helpers/index.js';
	import { errorMessage, isLoading } from '../../../stores/uiState';

	import { breedImagePath, breedLabel } from './breeds';
	import { loadBreedImageCredits, type BreedImageCredit } from './image-credits';
	import { iterate } from './handlers';
	import type { Animal, Candidate, ProblemInfo } from './types';

	const { data } = $props<{ data: { problems: ProblemInfo[] } }>();

	/** The demo looks its problems up by the names given in the seeding script. */
	const PROBLEM_NAMES: Record<Animal, string> = {
		cat: 'Cat breeds',
		dog: 'Dog breeds'
	};

	type Stage = 'choose' | 'explore' | 'result';

	let stage: Stage = $state('choose');
	let animal: Animal = $state('cat');
	let problem: ProblemInfo | null = $state(null);

	let referencePoint: number[] = $state([]);
	let lastIteratedPoint: number[] = $state([]);
	let candidates: Candidate[] = $state([]);
	let selectedIndex = $state(0);
	let iterationCount = $state(0);
	let hasIterated = $state(false);

	let imageCredits: Record<string, BreedImageCredit> = $state({});
	let imageFailed = $state(false);

	let problemList = $derived(data.problems ?? []);
	let selectedCandidate = $derived(candidates[selectedIndex] ?? null);
	let displayAccuracy = $derived(getDisplayAccuracy(problem));
	let selectedCredit = $derived(
		selectedCandidate ? (imageCredits[selectedCandidate.breedName] ?? null) : null
	);

	/**
	 * The objective values the preference bars are drawn against. Before the
	 * first iteration there is no solution to compare to, so the midpoint
	 * between the ideal and the nadir is used.
	 */
	let comparisonValues = $derived.by(() => {
		if (!problem) return [];
		if (selectedCandidate) {
			return problem.objectives.map(
				(objective) => selectedCandidate.objectiveValues[objective.symbol] ?? 0
			);
		}
		return problem.objectives.map((objective) => ((objective.ideal ?? 0) + (objective.nadir ?? 0)) / 2);
	});

	function midpoint(target: ProblemInfo): number[] {
		return target.objectives.map((objective) => ((objective.ideal ?? 0) + (objective.nadir ?? 0)) / 2);
	}

	function chooseAnimal(chosen: Animal) {
		const found = problemList.find((candidate: ProblemInfo) => candidate.name === PROBLEM_NAMES[chosen]);

		if (!found) {
			errorMessage.set(
				`The ${chosen} breed problem was not found. Seed it with desdeo/api/db_init_catsanddogs.py.`
			);
			return;
		}

		animal = chosen;
		problem = found;
		referencePoint = midpoint(found);
		lastIteratedPoint = [];
		candidates = [];
		selectedIndex = 0;
		iterationCount = 0;
		hasIterated = false;
		stage = 'explore';
	}

	function handlePreferenceChange(update: { preferenceValues: number[] }) {
		referencePoint = [...update.preferenceValues];
	}

	async function handleIterate(update: { preferenceValues: number[] }) {
		if (!problem) return;

		const point = [...update.preferenceValues];
		const found = await iterate(problem, animal, point);

		if (found === null) return;

		if (found.length === 0) {
			errorMessage.set('No breeds were found for this wish. Try changing it a little.');
			return;
		}

		referencePoint = point;
		lastIteratedPoint = point;
		candidates = found;
		selectedIndex = 0;
		imageFailed = false;
		iterationCount += 1;
		hasIterated = true;
	}

	function handleSelectSolution(index: number) {
		selectedIndex = index;
		imageFailed = false;
	}

	function restart() {
		stage = 'choose';
		problem = null;
		candidates = [];
		selectedIndex = 0;
		iterationCount = 0;
		hasIterated = false;
		errorMessage.set(null);
	}

	onMount(async () => {
		errorMessage.set(null);
		imageCredits = await loadBreedImageCredits();
	});
</script>

<svelte:head>
	<title>Cats and dogs | DESDEO</title>
</svelte:head>

{#if $isLoading}
	<LoadingSpinner text={animal === 'cat' ? 'Looking for cats...' : 'Looking for dogs...'} />
{/if}
{#if $errorMessage}
	<Alert title="Something went wrong" variant="destructive" />
{/if}

{#if stage === 'choose'}
	<div class="flex min-h-[calc(100vh-3rem)] items-center justify-center p-6">
		<Card.Root class="w-full max-w-3xl">
			<Card.Header>
				<Card.Title class="text-2xl">Find your ideal cat or dog breed</Card.Title>
				<Card.Description>
					Multiobjective optimization problems turn up in the unlikeliest places. The data behind
					this demo comes from two real survey studies on the behaviour of cat and dog breeds in
					Finland, but the problem formulation should not be taken too seriously. Whatever the
					method suggests, you are the decision maker and you make the final call.
				</Card.Description>
			</Card.Header>
			<Card.Content>
				<p class="mb-4 text-sm text-gray-600">I like...</p>
				<div class="grid gap-4 sm:grid-cols-2">
					<Button class="h-24 text-lg" onclick={() => chooseAnimal('cat')}>Cats 😻</Button>
					<Button class="h-24 text-lg" variant="secondary" onclick={() => chooseAnimal('dog')}>
						Dogs 🐕
					</Button>
				</div>
			</Card.Content>
		</Card.Root>
	</div>
{:else if stage === 'explore' && problem}
	<BaseLayout showLeftSidebar={true} showRightSidebar={false} bottomPanelTitle="Suggested breeds">
		{#snippet leftSidebar()}
			<AppSidebar
				problem={problem!}
				preferenceTypes={[PREFERENCE_TYPES.ReferencePoint]}
				typePreferences={PREFERENCE_TYPES.ReferencePoint}
				preferenceValues={referencePoint}
				objectiveValues={comparisonValues}
				lastIteratedPreference={lastIteratedPoint}
				onPreferenceChange={handlePreferenceChange}
				onIterate={handleIterate}
				isFinishButton={false}
			/>
		{/snippet}

		{#snippet explorerTitle()}
			<span>
				{animal === 'cat' ? 'Your ideal cat 😺' : 'Your ideal dog 🐶'}
				{#if iterationCount > 0}
					<span class="ml-2 text-sm font-normal text-gray-500">round {iterationCount}</span>
				{/if}
			</span>
		{/snippet}

		{#snippet explorerControls()}
			<Button variant="ghost" size="sm" onclick={restart}>Start over</Button>
			<Button size="sm" disabled={!selectedCandidate} onclick={() => (stage = 'result')}>
				This is the one
			</Button>
		{/snippet}

		{#snippet visualizationArea()}
			{#if hasIterated}
				<VisualizationsPanel
					{problem}
					previousPreferenceValues={[lastIteratedPoint]}
					currentPreferenceValues={referencePoint}
					previousPreferenceType={PREFERENCE_TYPES.ReferencePoint}
					currentPreferenceType={PREFERENCE_TYPES.ReferencePoint}
					solutionsObjectiveValues={candidates.map((candidate) =>
						problem!.objectives.map((objective) => candidate.objectiveValues[objective.symbol] ?? 0)
					)}
					externalSelectedIndexes={[selectedIndex]}
					lineLabels={Object.fromEntries(
						candidates.map((candidate, index) => [index, breedLabel(candidate.breedName)])
					)}
					onSelectSolution={handleSelectSolution}
				/>
			{:else}
				<div class="flex h-full flex-col items-center justify-center gap-2 text-center text-gray-600">
					<p class="text-lg font-medium">
						Describe the {animal} you would like
					</p>
					<p class="max-w-md text-sm">
						Set each trait on the left to the value you wish for, then press Iterate. You will get
						a handful of breeds that come as close to your wishes as the survey data allows.
					</p>
				</div>
			{/if}
		{/snippet}

		{#snippet numericalValues()}
			{#if candidates.length > 0}
				<div class="h-full overflow-auto">
					<Table.Root>
						<Table.Header>
							<Table.Row>
								<Table.Head>Breed group</Table.Head>
								{#each problem!.objectives as objective}
									<Table.Head class="whitespace-nowrap">
										{objective.name}
										<span class="text-gray-500">({objective.maximize ? 'max' : 'min'})</span>
									</Table.Head>
								{/each}
							</Table.Row>
						</Table.Header>
						<Table.Body>
							{#each candidates as candidate, index}
								<Table.Row
									class={index === selectedIndex ? 'bg-gray-100 font-medium' : 'cursor-pointer'}
									onclick={() => (selectedIndex = index)}
								>
									<Table.Cell class="whitespace-nowrap">
										{breedLabel(candidate.breedName)}
									</Table.Cell>
									{#each problem!.objectives as objective, objectiveIndex}
										<Table.Cell>
											{formatNumber(
												candidate.objectiveValues[objective.symbol] ?? 0,
												displayAccuracy[objectiveIndex]
											)}
										</Table.Cell>
									{/each}
								</Table.Row>
							{/each}
						</Table.Body>
					</Table.Root>
				</div>
			{:else}
				<div class="p-4 text-sm text-gray-600">
					No breeds suggested yet. Set your wishes and press Iterate.
				</div>
			{/if}
		{/snippet}
	</BaseLayout>
{:else if stage === 'result' && problem && selectedCandidate}
	<div class="flex min-h-[calc(100vh-3rem)] items-center justify-center p-6">
		<Card.Root class="w-full max-w-2xl">
			<Card.Header>
				<Card.Title class="text-2xl">
					{animal === 'cat' ? 'Your ideal cat breed is' : 'Your ideal dog breed is'}
					{breedLabel(selectedCandidate.breedName)}
				</Card.Title>
				<Card.Description>
					Found after {iterationCount}
					{iterationCount === 1 ? 'round' : 'rounds'} of the reference point method.
				</Card.Description>
			</Card.Header>
			<Card.Content class="flex flex-col gap-6">
				{#if !imageFailed}
					<figure class="flex flex-col items-center gap-2">
						<img
							class="max-h-80 rounded-lg object-contain"
							src={breedImagePath(selectedCandidate.breedName)}
							alt={breedLabel(selectedCandidate.breedName)}
							onerror={() => (imageFailed = true)}
						/>
						{#if selectedCredit}
							<figcaption class="text-center text-xs text-gray-500">
								{#if selectedCredit.represented_by !== breedLabel(selectedCandidate.breedName)}
									Pictured: {selectedCredit.represented_by}.
								{/if}
								Photograph by {selectedCredit.author}, {selectedCredit.license},
								<a class="underline" href={selectedCredit.source} target="_blank" rel="noreferrer">
									via Wikimedia Commons
								</a>.
							</figcaption>
						{/if}
					</figure>
				{/if}

				<Table.Root>
					<Table.Header>
						<Table.Row>
							<Table.Head>Trait</Table.Head>
							<Table.Head class="text-right">Value</Table.Head>
						</Table.Row>
					</Table.Header>
					<Table.Body>
						{#each problem.objectives as objective, objectiveIndex}
							<Table.Row>
								<Table.Cell>
									{objective.name}
									<span class="text-gray-500">({objective.maximize ? 'max' : 'min'})</span>
								</Table.Cell>
								<Table.Cell class="text-right">
									{formatNumber(
										selectedCandidate.objectiveValues[objective.symbol] ?? 0,
										displayAccuracy[objectiveIndex]
									)}
								</Table.Cell>
							</Table.Row>
						{/each}
					</Table.Body>
				</Table.Root>

				<p class="text-sm text-gray-600">
					Remember that this is a playful example. A real breed choice deserves rather more thought
					than seven numbers from a survey.
				</p>
			</Card.Content>
			<Card.Footer class="flex gap-2">
				<Button variant="secondary" onclick={() => (stage = 'explore')}>Keep looking</Button>
				<Button onclick={restart}>Start over</Button>
			</Card.Footer>
		</Card.Root>
	</div>
{/if}
