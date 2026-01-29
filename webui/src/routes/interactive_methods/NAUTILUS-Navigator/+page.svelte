<script lang="ts">
	import { onMount } from 'svelte';
	import { methodSelection } from '../../../stores/methodSelection';
	import type { MethodSelectionState } from '../../../stores/methodSelection';
	import { isLoading, errorMessage } from '../../../stores/uiState';

	import BaseLayout from '$lib/components/custom/method_layout/base-layout.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import Input from '$lib/components/ui/input/input.svelte';
	import Checkbox from '$lib/components/ui/checkbox/checkbox.svelte';
	import Alert from '$lib/components/custom/notifications/alert.svelte';
	import RangeTrajectoryChart from '$lib/components/visualizations/range-trajectory/range-trajectory.svelte';

	import type {
		NautilusNavigatorInitializeResponse,
		NautilusNavigatorNavigateResponse,
		ProblemGetRequest,
		ProblemInfo
	} from '$lib/gen/models';
	import {
		fetch_problem_info,
		initialize_navigator,
		navigate_navigator
	} from './handler';

	let selection = $state<MethodSelectionState>({ selectedProblemId: null, selectedMethod: null });
	let problem_info = $state<ProblemInfo | null>(null);
	let init_response = $state<NautilusNavigatorInitializeResponse | null>(null);
	let navigation_response = $state<NautilusNavigatorNavigateResponse | null>(null);

	let total_steps = $state(10);
	let go_back_step = $state(0);
	let reference_point = $state<Record<string, number>>({});
	let bounds = $state<Record<string, number>>({});
	let use_bounds = $state(false);
	let defaults_ready = $state(false);
	const totalStepsId = 'nautilus-total-steps';
	const goBackId = 'nautilus-go-back-step';
	const boundsToggleId = 'nautilus-enable-bounds';

	let objective_keys = $derived.by(() => {
		if (!problem_info?.objectives) return [];
		return problem_info.objectives.map((obj) => obj.symbol);
	});

	let objective_labels = $derived.by(() => {
		if (!problem_info?.objectives) return objective_keys;
		return problem_info.objectives.map((obj) => obj.name || obj.symbol);
	});

	let step_numbers = $derived.by(() => {
		if (navigation_response?.step_numbers?.length) {
			return navigation_response.step_numbers;
		}
		if (init_response) {
			return [init_response.step_number];
		}
		return [];
	});

	function getObjectiveSeries(symbol: string) {
		if (navigation_response) {
			return {
				lower: navigation_response.lower_bounds[symbol] ?? [],
				upper: navigation_response.upper_bounds[symbol] ?? [],
				navigation: navigation_response.navigation_points[symbol] ?? [],
				preferences: navigation_response.preferences[symbol] ?? [],
				bounds: navigation_response.bounds[symbol] ?? []
			};
		}

		if (init_response) {
			return {
				lower: [init_response.reachable_bounds.lower_bounds[symbol]],
				upper: [init_response.reachable_bounds.upper_bounds[symbol]],
				navigation: [init_response.navigation_point[symbol]],
				preferences: [],
				bounds: []
			};
		}

		return { lower: [], upper: [], navigation: [], preferences: [], bounds: [] };
	}

	function alignPreference(series: number[], targetLength: number): Array<number | null> {
		const aligned: Array<number | null> = [];
		for (let i = 0; i < targetLength; i += 1) {
			if (i === 0) {
				aligned.push(null);
			} else {
				aligned.push(series[i - 1] ?? null);
			}
		}
		return aligned;
	}

	function updateReference(symbol: string, value: string) {
		const parsed = Number(value);
		reference_point = { ...reference_point, [symbol]: parsed };
	}

	function updateBounds(symbol: string, value: string) {
		const parsed = Number(value);
		bounds = { ...bounds, [symbol]: parsed };
	}

	async function handle_initialize() {
		if (selection.selectedProblemId === null) {
			errorMessage.set('No problem selected for NAUTILUS Navigator.');
			return;
		}

		try {
			isLoading.set(true);
			errorMessage.set('');

			const response = await initialize_navigator({
				problem_id: selection.selectedProblemId,
				session_id: selection.selectedSessionId ?? undefined,
				total_steps
			});

			if (!response) {
				errorMessage.set('Failed to initialize NAUTILUS Navigator.');
				return;
			}

			init_response = response;
			navigation_response = null;
			go_back_step = response.step_number;
			defaults_ready = false;
		} catch (err) {
			console.error('Error during NAUTILUS Navigator initialization', err);
			errorMessage.set('Unexpected error during NAUTILUS Navigator initialization.');
		} finally {
			isLoading.set(false);
		}
	}

	async function handle_navigate() {
		if (selection.selectedProblemId === null) {
			errorMessage.set('No problem selected for NAUTILUS Navigator.');
			return;
		}

		const steps_remaining = Math.max(total_steps - go_back_step, 0);

		if (steps_remaining === 0) {
			errorMessage.set('No steps remaining. Increase total steps or go back to an earlier step.');
			return;
		}

		try {
			isLoading.set(true);
			errorMessage.set('');

			const response = await navigate_navigator({
				problem_id: selection.selectedProblemId,
				session_id: selection.selectedSessionId ?? undefined,
				total_steps,
				go_back_step,
				steps_remaining: 1,
				reference_point,
				bounds: use_bounds ? bounds : null
			});

			if (!response) {
				errorMessage.set('Failed to navigate NAUTILUS Navigator.');
				return;
			}

			navigation_response = response;
			go_back_step = response.current_step;
		} catch (err) {
			console.error('Error during NAUTILUS Navigator navigation', err);
			errorMessage.set('Unexpected error during NAUTILUS Navigator navigation.');
		} finally {
			isLoading.set(false);
		}
	}

	onMount(() => {
		const unsubscribe = methodSelection.subscribe((v) => (selection = v));

		(async () => {
			if (selection.selectedProblemId === null) {
				console.log('No problem selected for NAUTILUS Navigator.');
				return;
			}

			try {
				isLoading.set(true);
				const request: ProblemGetRequest = { problem_id: selection.selectedProblemId };
				const response = await fetch_problem_info(request);
				if (!response) {
					errorMessage.set('Could not fetch problem information for NAUTILUS Navigator.');
					return;
				}
				problem_info = response;
				await handle_initialize();
			} catch (err) {
				console.error('Error during NAUTILUS Navigator setup', err);
				errorMessage.set('Unexpected error during NAUTILUS Navigator setup.');
			} finally {
				isLoading.set(false);
			}
		})();

		return unsubscribe;
	});

	$effect(() => {
		if (!defaults_ready && init_response && problem_info?.objectives) {
			const defaults: Record<string, number> = {};
			const boundsDefaults: Record<string, number> = {};
			problem_info.objectives.forEach((obj) => {
				const lower = init_response.reachable_bounds.lower_bounds[obj.symbol];
				const upper = init_response.reachable_bounds.upper_bounds[obj.symbol];
				defaults[obj.symbol] = (lower + upper) / 2;
				boundsDefaults[obj.symbol] = upper;
			});
			reference_point = defaults;
			bounds = boundsDefaults;
			defaults_ready = true;
		}
	});
</script>

<h1 class="mt-10 text-center text-2xl font-semibold">NAUTILUS Navigator</h1>

{#if $isLoading}
	<p class="text-center text-gray-500">Loading NAUTILUS Navigator data…</p>
{/if}

{#if $errorMessage}
	<div class="mx-auto mt-4 max-w-3xl">
		<Alert message={$errorMessage} type="error" onDismiss={() => errorMessage.set('')} />
	</div>
{/if}

{#if problem_info}
	<BaseLayout
		showRightSidebar={false}
		bottomPanelTitle="Navigation summary"
		leftSidebarWidth="320px"
	>
		{#snippet leftSidebar()}
			<div class="flex h-full flex-col gap-4 p-4">
				<div>
					<h2 class="text-lg font-semibold">Problem</h2>
					<p class="text-sm text-gray-500">{problem_info.name}</p>
				</div>

				<div class="space-y-2 rounded border p-3">
						<label class="text-sm font-medium text-gray-700" for={totalStepsId}>Total steps</label>
						<Input
							id={totalStepsId}
							type="number"
							min="1"
							value={total_steps}
							on:input={(event) => (total_steps = Number(event.currentTarget.value))}
						/>
					<Button class="mt-2 w-full" onclick={handle_initialize}>Restart</Button>
				</div>

				<div class="space-y-3 rounded border p-3">
					<div class="flex items-center justify-between">
							<label class="text-sm font-medium text-gray-700" for={boundsToggleId}>Enable bounds</label>
							<Checkbox id={boundsToggleId} bind:checked={use_bounds} />
						</div>

						<label class="text-sm font-medium text-gray-700" for={goBackId}>Go back to step</label>
						<Input
							id={goBackId}
							type="number"
							min="0"
							max={total_steps}
							value={go_back_step}
							on:input={(event) => (go_back_step = Number(event.currentTarget.value))}
					/>
				</div>

				<div class="space-y-4">
					<h3 class="text-sm font-semibold text-gray-700">Preferences</h3>
					{#each objective_keys as symbol, index}
						<div class="rounded border p-3">
							<p class="text-sm font-medium text-gray-700">{objective_labels[index]}</p>
								<label class="mt-2 block text-xs text-gray-500" for={`nautilus-ref-${symbol}`}>
									Reference point
								</label>
								<Input
									id={`nautilus-ref-${symbol}`}
									type="number"
									value={reference_point[symbol] ?? ''}
									on:input={(event) => updateReference(symbol, event.currentTarget.value)}
								/>
								<label class="mt-2 block text-xs text-gray-500" for={`nautilus-bound-${symbol}`}>
									Bound
								</label>
								<Input
									id={`nautilus-bound-${symbol}`}
									type="number"
									value={bounds[symbol] ?? ''}
									disabled={!use_bounds}
								on:input={(event) => updateBounds(symbol, event.currentTarget.value)}
							/>
						</div>
					{/each}
				</div>

				<Button class="w-full" onclick={handle_navigate}>Navigate</Button>
			</div>
		{/snippet}

		{#snippet visualizationArea()}
			{#if step_numbers.length === 0}
				<div class="flex h-full items-center justify-center text-gray-500">
					No navigation data available yet.
				</div>
			{:else}
				<div class="grid gap-4 xl:grid-cols-2">
					{#each objective_keys as symbol, index}
						{#key symbol}
							{#if step_numbers.length > 0}
								{@const series = getObjectiveSeries(symbol)}
								{@const preferenceSeries = alignPreference(series.preferences, step_numbers.length)}
								{@const boundSeries = alignPreference(series.bounds, step_numbers.length)}
								<RangeTrajectoryChart
									label={objective_labels[index]}
									steps={step_numbers}
									lowerBounds={series.lower}
									upperBounds={series.upper}
									navigationPoints={series.navigation}
									referencePoints={preferenceSeries}
									boundPoints={boundSeries}
								/>
							{/if}
						{/key}
					{/each}
				</div>
			{/if}
		{/snippet}

		{#snippet numericalValues()}
			<div class="overflow-auto rounded border bg-white p-4">
				{#if navigation_response}
					<p class="text-sm text-gray-600">
						Current step: {navigation_response.current_step} / {navigation_response.total_steps}
					</p>
					<table class="mt-3 w-full text-sm">
						<thead>
							<tr class="border-b text-left text-gray-500">
								<th class="py-2">Objective</th>
								<th class="py-2">Lower bound</th>
								<th class="py-2">Upper bound</th>
								<th class="py-2">Navigation point</th>
							</tr>
						</thead>
						<tbody>
							{#each objective_keys as symbol, index}
								{@const series = getObjectiveSeries(symbol)}
								<tr class="border-b">
									<td class="py-2">{objective_labels[index]}</td>
									<td class="py-2">{series.lower.at(-1)}</td>
									<td class="py-2">{series.upper.at(-1)}</td>
									<td class="py-2">{series.navigation.at(-1)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				{:else if init_response}
					<p class="text-sm text-gray-600">Initialized at step {init_response.step_number}.</p>
				{:else}
					<p class="text-sm text-gray-500">No navigation summary available.</p>
				{/if}
			</div>
		{/snippet}
	</BaseLayout>
{:else}
	{#if !$isLoading}
		<div class="mt-10 text-center text-gray-500">Select a problem to start NAUTILUS Navigator.</div>
	{/if}
{/if}
