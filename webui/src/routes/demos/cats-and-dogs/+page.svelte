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

	import { SegmentedControl } from '$lib/components/custom/segmented-control';

	import { breedImagePath, breedLabel } from './breeds';
	import { loadBreedImageCredits, type BreedImageCredit } from './image-credits';
	import { iterate } from './handlers';
	import {
		LANGUAGES,
		LANGUAGE_NAMES,
		OBJECTIVE_NAMES,
		TRANSLATIONS,
		storeLanguage,
		storedLanguage,
		type Language
	} from './i18n';
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

	// SegmentedControl binds a plain string, so the choice is held as one and
	// narrowed to a Language for the lookups.
	let languageChoice = $state<string>(storedLanguage());
	let language: Language = $derived(languageChoice === 'fi' ? 'fi' : 'en');
	let t = $derived(TRANSLATIONS[language]);

	const languageOptions = LANGUAGES.map((code) => ({ value: code, label: LANGUAGE_NAMES[code] }));

	$effect(() => {
		storeLanguage(language);
	});

	/**
	 * The problem with its objectives renamed into the chosen language. The
	 * shared sidebar, the visualizations and the tables all read the names off
	 * the problem, so translating them here is what translates all three.
	 */
	let localizedProblem = $derived.by(() => {
		if (!problem) return null;
		const names = OBJECTIVE_NAMES[language];
		return {
			...problem,
			objectives: problem.objectives.map((objective) => ({
				...objective,
				name: names[objective.symbol] ?? objective.name
			}))
		};
	});

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
			errorMessage.set(t.problemMissing(chosen));
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
			errorMessage.set(t.noBreedsFound);
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
	<LoadingSpinner text={t.searching[animal]} />
{/if}
{#if $errorMessage}
	<Alert title={t.errorTitle} variant="destructive" />
{/if}

{#if stage === 'choose'}
	<div class="flex min-h-[calc(100vh-3rem)] items-center justify-center p-6">
		<Card.Root class="w-full max-w-3xl">
			<Card.Header>
				<div class="flex items-start justify-between gap-4">
					<Card.Title class="text-2xl">{t.introTitle}</Card.Title>
					<SegmentedControl size="sm" options={languageOptions} bind:value={languageChoice} />
				</div>
				<Card.Description>{t.introBody}</Card.Description>
			</Card.Header>
			<Card.Content>
				<p class="mb-4 text-sm text-gray-600">{t.introPrompt}</p>
				<div class="grid gap-4 sm:grid-cols-2">
					<Button class="h-24 text-lg" onclick={() => chooseAnimal('cat')}>{t.chooseCats}</Button>
					<Button class="h-24 text-lg" variant="secondary" onclick={() => chooseAnimal('dog')}>
						{t.chooseDogs}
					</Button>
				</div>
			</Card.Content>
		</Card.Root>
	</div>
{:else if stage === 'explore' && problem}
	<BaseLayout showLeftSidebar={true} showRightSidebar={false} bottomPanelTitle={t.suggestedBreeds}>
		{#snippet leftSidebar()}
			<AppSidebar
				problem={localizedProblem!}
				preferenceTypes={[PREFERENCE_TYPES.ReferencePoint]}
				typePreferences={PREFERENCE_TYPES.ReferencePoint}
				preferenceValues={referencePoint}
				objectiveValues={comparisonValues}
				lastIteratedPreference={lastIteratedPoint}
				onPreferenceChange={handlePreferenceChange}
				onIterate={handleIterate}
				isFinishButton={false}
				labels={t.sidebar}
			/>
		{/snippet}

		{#snippet explorerTitle()}
			<span>
				{t.explorerTitle[animal]}
				{#if iterationCount > 0}
					<span class="ml-2 text-sm font-normal text-gray-500">{t.round(iterationCount)}</span>
				{/if}
			</span>
		{/snippet}

		{#snippet explorerControls()}
			<SegmentedControl size="sm" options={languageOptions} bind:value={languageChoice} />
			<Button variant="ghost" size="sm" onclick={restart}>{t.startOver}</Button>
			<Button size="sm" disabled={!selectedCandidate} onclick={() => (stage = 'result')}>
				{t.chooseThis}
			</Button>
		{/snippet}

		{#snippet visualizationArea()}
			{#if hasIterated}
				<VisualizationsPanel
					problem={localizedProblem}
					previousPreferenceValues={[lastIteratedPoint]}
					currentPreferenceValues={referencePoint}
					previousPreferenceType={PREFERENCE_TYPES.ReferencePoint}
					currentPreferenceType={PREFERENCE_TYPES.ReferencePoint}
					solutionsObjectiveValues={candidates.map((candidate) =>
						problem!.objectives.map((objective) => candidate.objectiveValues[objective.symbol] ?? 0)
					)}
					externalSelectedIndexes={[selectedIndex]}
					lineLabels={Object.fromEntries(
						candidates.map((_, index) => [index, t.candidate(index + 1)])
					)}
					onSelectSolution={handleSelectSolution}
					labels={t.visualization}
				/>
			{:else}
				<div class="flex h-full flex-col items-center justify-center gap-2 text-center text-gray-600">
					<p class="text-lg font-medium">{t.emptyTitle[animal]}</p>
					<p class="max-w-md text-sm">{t.emptyBody}</p>
				</div>
			{/if}
		{/snippet}

		{#snippet numericalValues()}
			{#if candidates.length > 0}
				<div class="h-full overflow-auto">
					<Table.Root>
						<Table.Header>
							<Table.Row>
								<Table.Head>{t.suggestedBreeds}</Table.Head>
								{#each localizedProblem!.objectives as objective}
									<Table.Head class="whitespace-nowrap">
										{objective.name}
										<span class="text-gray-500">({objective.maximize ? t.max : t.min})</span>
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
										{t.candidate(index + 1)}
									</Table.Cell>
									{#each localizedProblem!.objectives as objective, objectiveIndex}
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
				<div class="p-4 text-sm text-gray-600">{t.noBreedsYet}</div>
			{/if}
		{/snippet}
	</BaseLayout>
{:else if stage === 'result' && problem && selectedCandidate}
	<div class="flex min-h-[calc(100vh-3rem)] items-center justify-center p-6">
		<Card.Root class="w-full max-w-2xl">
			<Card.Header>
				<div class="flex items-start justify-between gap-4">
					<Card.Title class="text-2xl">
						{t.resultTitle[animal]}
						{breedLabel(selectedCandidate.breedName)}
					</Card.Title>
					<SegmentedControl size="sm" options={languageOptions} bind:value={languageChoice} />
				</div>
				<Card.Description>{t.foundAfter(iterationCount)}</Card.Description>
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
									{t.pictured(selectedCredit.represented_by)}
								{/if}
								{t.photographBy(selectedCredit.author, selectedCredit.license)}
								<a class="underline" href={selectedCredit.source} target="_blank" rel="noreferrer">
									{t.viaCommons}
								</a>.
							</figcaption>
						{/if}
					</figure>
				{/if}

				<Table.Root>
					<Table.Header>
						<Table.Row>
							<Table.Head>{t.trait}</Table.Head>
							<Table.Head class="text-right">{t.value}</Table.Head>
						</Table.Row>
					</Table.Header>
					<Table.Body>
						{#each localizedProblem!.objectives as objective, objectiveIndex}
							<Table.Row>
								<Table.Cell>
									{objective.name}
									<span class="text-gray-500">({objective.maximize ? t.max : t.min})</span>
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

				<p class="text-sm text-gray-600">{t.disclaimer}</p>
			</Card.Content>
			<Card.Footer class="flex gap-2">
				<Button variant="secondary" onclick={() => (stage = 'explore')}>{t.keepLooking}</Button>
				<Button onclick={restart}>{t.startOver}</Button>
			</Card.Footer>
		</Card.Root>
	</div>
{/if}
