<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { methodSelection } from '../../../stores/methodSelection';
	import type { MethodSelectionState } from '../../../stores/methodSelection';
	import { isLoading, errorMessage } from '../../../stores/uiState';

	import BaseLayout from '$lib/components/custom/method_layout/base-layout.svelte';
	import ReachableBands from '$lib/components/visualizations/nautilus-navigator/reachable-bands.svelte';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input/index.js';
	import { Slider } from '$lib/components/ui/slider/index.js';
	import * as Card from '$lib/components/ui/card';

	import type { ProblemGetRequest, ProblemInfo } from '$lib/gen/models';
	import {
		fetch_problem_info,
		initialize_navigator,
		recompute_navigator,
		type NautilusNavigatorSegmentResponse
	} from './handler';

	let selection = $state<MethodSelectionState>({
		selectedProblemId: null,
		selectedMethod: null,
		selectedSessionId: null,
		selectedSessionInfo: null
	});

	let problemInfo = $state<ProblemInfo | null>(null);
	let navigatorStateId = $state<number | null>(null);
	let sessionId = $state<number | null>(null);

	let objectiveSymbols = $state<string[]>([]);
	let lowerBounds = $state<Record<string, number[]>>({});
	let upperBounds = $state<Record<string, number[]>>({});
	let navigationPoints = $state<Record<string, number[]>>({});
	let distance = $state<number[]>([]);
	let totalSteps = $state(0);

	let aspirationLevels = $state<Record<string, number>>({});
	let boundsLevels = $state<Record<string, number | null>>({});

	let stepCursor = $state(0);
	let goBackStep = $state(0);
	let segmentSteps = $state(20);
	let playbackDelay = $state(400);
	let isPlaying = $state(false);

	let playbackTimer: ReturnType<typeof setInterval> | null = null;

	const objectiveLabels = $derived.by(() => {
		if (!problemInfo) return {};
		return Object.fromEntries(problemInfo.objectives.map((obj) => [obj.symbol, obj.name]));
	});

	const availableSteps = $derived.by(() => {
		if (objectiveSymbols.length === 0) return 0;
		const lengths = objectiveSymbols.map((symbol) => lowerBounds[symbol]?.length ?? 0);
		const minLength = lengths.length ? Math.min(...lengths) : 0;
		return Math.max(0, minLength - 1);
	});

	const currentMetrics = $derived.by(() => {
		const values: Record<string, { lower?: number; upper?: number; nav?: number }> = {};
		for (const symbol of objectiveSymbols) {
			values[symbol] = {
				lower: lowerBounds[symbol]?.[stepCursor],
				upper: upperBounds[symbol]?.[stepCursor],
				nav: navigationPoints[symbol]?.[stepCursor]
			};
		}
		return values;
	});

	$effect(() => {
		if (stepCursor > availableSteps) {
			stepCursor = availableSteps;
		}
		if (goBackStep > availableSteps) {
			goBackStep = availableSteps;
		}
	});

	$effect(() => {
		if (!isPlaying) {
			if (playbackTimer) {
				clearInterval(playbackTimer);
				playbackTimer = null;
			}
			return;
		}

		if (playbackTimer) {
			clearInterval(playbackTimer);
		}

		playbackTimer = setInterval(() => {
			if (stepCursor >= availableSteps) {
				isPlaying = false;
				return;
			}
			stepCursor += 1;
		}, playbackDelay);
	});

	onDestroy(() => {
		if (playbackTimer) {
			clearInterval(playbackTimer);
		}
	});

	function mergeSeries(existing: number[] | undefined, incoming: number[], start: number) {
		if (!existing || existing.length === 0 || start === 0) {
			return [...incoming];
		}
		return [...existing.slice(0, start), ...incoming];
	}

	function applySegment(segment: NautilusNavigatorSegmentResponse) {
		totalSteps = segment.total_steps;
		objectiveSymbols = segment.objective_symbols;

		for (const symbol of segment.objective_symbols) {
			lowerBounds = {
				...lowerBounds,
				[symbol]: mergeSeries(lowerBounds[symbol], segment.lower_bounds[symbol], segment.segment_start_step)
			};
			upperBounds = {
				...upperBounds,
				[symbol]: mergeSeries(upperBounds[symbol], segment.upper_bounds[symbol], segment.segment_start_step)
			};
			if (segment.navigation_points?.[symbol]) {
				navigationPoints = {
					...navigationPoints,
					[symbol]: mergeSeries(
						navigationPoints[symbol],
						segment.navigation_points[symbol],
						segment.segment_start_step
					)
				};
			}
		}

		if (segment.distance) {
			distance = mergeSeries(distance, segment.distance, segment.segment_start_step);
		}

		stepCursor = segment.segment_start_step;
		goBackStep = segment.segment_start_step;
	}

	function initializePreferences(segment: NautilusNavigatorSegmentResponse) {
		const newAspiration: Record<string, number> = {};
		const newBounds: Record<string, number | null> = {};

		for (const symbol of segment.objective_symbols) {
			const objective = problemInfo?.objectives.find((obj) => obj.symbol === symbol);
			const navStart = segment.navigation_points?.[symbol]?.[0];
			newAspiration[symbol] = objective?.ideal ?? navStart ?? objective?.nadir ?? 0;
			newBounds[symbol] = null;
		}

		aspirationLevels = newAspiration;
		boundsLevels = newBounds;
	}

	async function loadInitialState(problemId: number) {
		isLoading.set(true);

		try {
			const problemRequest: ProblemGetRequest = { problem_id: problemId };
			const problemResponse = await fetch_problem_info(problemRequest);

			if (!problemResponse) {
				errorMessage.set('Failed to load problem information for NAUTILUS Navigator.');
				return;
			}

			problemInfo = problemResponse;

			const initResponse = await initialize_navigator({
				problem_id: problemId,
				session_id: selection.selectedSessionId ?? undefined,
				total_steps: totalSteps || 100
			});

			if (!initResponse) {
				errorMessage.set('Failed to initialize NAUTILUS Navigator.');
				return;
			}

			navigatorStateId = initResponse.state_id;
			sessionId = initResponse.session_id;
			applySegment(initResponse);
			initializePreferences(initResponse);
		} finally {
			isLoading.set(false);
		}
	}

	async function handleRecompute(targetStep: number) {
		if (!selection.selectedProblemId) {
			errorMessage.set('No problem selected.');
			return;
		}
		if (!navigatorStateId) {
			errorMessage.set('NAUTILUS Navigator has not been initialized.');
			return;
		}

		isPlaying = false;
		isLoading.set(true);

		try {
			const response = await recompute_navigator({
				problem_id: selection.selectedProblemId,
				session_id: sessionId ?? undefined,
				parent_state_id: navigatorStateId,
				go_back_step: targetStep,
				steps: segmentSteps,
				reference_point: aspirationLevels,
				bounds: boundsLevels
			});

			if (!response) {
				errorMessage.set('Failed to recompute NAUTILUS Navigator segment.');
				return;
			}

			navigatorStateId = response.state_id;
			sessionId = response.session_id;
			applySegment(response);
		} finally {
			isLoading.set(false);
		}
	}

	onMount(() => {
		const unsubscribe = methodSelection.subscribe((value) => (selection = value));

		if (selection.selectedProblemId) {
			loadInitialState(selection.selectedProblemId);
		}

		return unsubscribe;
	});
</script>

<BaseLayout
	leftSidebarWidth="320px"
	rightSidebarWidth="320px"
	showRightSidebar={false}
	bottomPanelTitle="Current step details"
>
	{#snippet visualizationArea()}
		<ReachableBands
			objectiveSymbols={objectiveSymbols}
			objectiveLabels={objectiveLabels}
			lowerBounds={lowerBounds}
			upperBounds={upperBounds}
			navigationPoints={navigationPoints}
			aspirationLevels={aspirationLevels}
			stepCursor={stepCursor}
			on:stepSelect={(event) => {
				goBackStep = event.detail;
				handleRecompute(event.detail);
			}}
		/>
	{/snippet}

	{#snippet explorerTitle()}
		<span>NAUTILUS Navigator</span>
	{/snippet}

	{#snippet explorerControls()}
		<div class="flex items-center gap-2">
			<Button size="sm" variant="outline" on:click={() => (isPlaying = !isPlaying)}>
				{isPlaying ? 'Pause' : 'Play'}
			</Button>
			<div class="min-w-[140px]">
				<Slider
					min={200}
					max={1200}
					step={50}
					value={[playbackDelay]}
					onValueChange={(value) => (playbackDelay = value[0])}
				/>
			</div>
			<span class="text-xs text-muted-foreground">{playbackDelay} ms/step</span>
		</div>
	{/snippet}

	{#snippet leftSidebar()}
		<div class="flex flex-col gap-4 p-3">
			<Card.Root>
				<Card.Header>
					<Card.Title>Aspiration levels</Card.Title>
					<Card.Description>Set your desired aspiration per objective.</Card.Description>
				</Card.Header>
				<Card.Content class="space-y-3">
					{#each objectiveSymbols as symbol}
						<label class="space-y-1 text-sm">
							<span class="font-medium">{objectiveLabels[symbol] ?? symbol}</span>
							<Input
								type="number"
								value={aspirationLevels[symbol] ?? ''}
								on:input={(event) => {
									const value = Number((event.target as HTMLInputElement).value);
									aspirationLevels = { ...aspirationLevels, [symbol]: value };
								}}
							/>
						</label>
					{/each}
				</Card.Content>
			</Card.Root>

			<Card.Root>
				<Card.Header>
					<Card.Title>Optional bounds</Card.Title>
					<Card.Description>Leave blank to remove bounds.</Card.Description>
				</Card.Header>
				<Card.Content class="space-y-3">
					{#each objectiveSymbols as symbol}
						<label class="space-y-1 text-sm">
							<span class="font-medium">{objectiveLabels[symbol] ?? symbol}</span>
							<Input
								type="number"
								value={boundsLevels[symbol] ?? ''}
								placeholder="unset"
								on:input={(event) => {
									const raw = (event.target as HTMLInputElement).value;
									boundsLevels = {
										...boundsLevels,
										[symbol]: raw === '' ? null : Number(raw)
									};
								}}
							/>
						</label>
					{/each}
				</Card.Content>
			</Card.Root>

			<Card.Root>
				<Card.Header>
					<Card.Title>Recompute</Card.Title>
					<Card.Description>Compute a new segment from a selected step.</Card.Description>
				</Card.Header>
				<Card.Content class="space-y-4">
					<div class="space-y-2">
						<div class="flex items-center justify-between text-xs text-muted-foreground">
							<span>Go back step</span>
							<span>{goBackStep + 1}</span>
						</div>
						<Slider
							min={0}
							max={availableSteps}
							step={1}
							value={[goBackStep]}
							onValueChange={(value) => (goBackStep = value[0])}
						/>
					</div>

					<label class="space-y-1 text-sm">
						<span class="font-medium">Segment length</span>
						<Input
							type="number"
							min="1"
							value={segmentSteps}
							on:input={(event) => (segmentSteps = Number((event.target as HTMLInputElement).value))}
						/>
					</label>

					<Button size="sm" on:click={() => handleRecompute(goBackStep)}>
						Recompute segment
					</Button>
				</Card.Content>
			</Card.Root>
		</div>
	{/snippet}

	{#snippet numericalValues()}
		<div class="p-4">
			<div class="mb-3 text-sm font-semibold">Step {stepCursor + 1}</div>
			<div class="grid gap-3 text-sm">
				{#each objectiveSymbols as symbol}
					<div class="rounded border p-2">
						<div class="font-medium">{objectiveLabels[symbol] ?? symbol}</div>
						<div class="mt-1 text-xs text-muted-foreground">
							Lower: {currentMetrics[symbol]?.lower?.toFixed(4) ?? '—'} | Upper:
							{currentMetrics[symbol]?.upper?.toFixed(4) ?? '—'} | Navigation:
							{currentMetrics[symbol]?.nav?.toFixed(4) ?? '—'}
						</div>
					</div>
				{/each}
			</div>
		</div>
	{/snippet}
</BaseLayout>
