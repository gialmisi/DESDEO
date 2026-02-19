<script lang="ts">
	import { onMount } from 'svelte';
	import { methodSelection } from '../../../stores/methodSelection';
	import type { MethodSelectionState } from '../../../stores/methodSelection';
	import type {
		ProblemGetRequest,
		ProblemInfo,
		InteractiveSessionBase,
		ScenarioRPMState,
		ScenarioComparisonResponse,
		ClusterComparison,
		SolverResults
	} from '$lib/gen/models';
	import { isLoading, errorMessage } from '../../../stores/uiState';

	import BaseLayout from '$lib/components/custom/method_layout/base-layout.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import VisualizationsPanel from '$lib/components/custom/visualizations-panel/visualizations-panel.svelte';
	import * as Tabs from '$lib/components/ui/tabs';
	import Honeycomb from '$lib/components/custom/honeycomb/honeycomb.svelte';
	import { DISTANCE_GRADIENT_CSS, CLUSTER_BORDER_COLORS } from '$lib/components/custom/honeycomb/honeycomb-utils';
	import HorizontalBar from '$lib/components/visualizations/horizontal-bar/horizontal-bar.svelte';
	import ParallelCoordinates from '$lib/components/visualizations/parallel-coordinates/parallel-coordinates.svelte';
	import Combobox from '$lib/components/ui/combobox/combobox.svelte';
	import Input from '$lib/components/ui/input/input.svelte';
	import Label from '$lib/components/ui/label/label.svelte';
	import Textarea from '$lib/components/ui/textarea/textarea.svelte';

	import {
		type ColumnDef,
		type Column,
		type SortingState,
		getCoreRowModel,
		getSortedRowModel
	} from '@tanstack/table-core';
	import { createSvelteTable } from '$lib/components/ui/data-table/data-table.svelte.js';
	import FlexRender from '$lib/components/ui/data-table/flex-render.svelte';
	import * as Table from '$lib/components/ui/table/index.js';
	import { renderSnippet } from '$lib/components/ui/data-table/render-helpers.js';

	import { COLOR_PALETTE } from '$lib/components/visualizations/utils/colors.ts';
	import { getDisplayAccuracy, formatNumber } from '$lib/helpers';
	import ArrowUpIcon from '@lucide/svelte/icons/arrow-up';
	import ArrowDownIcon from '@lucide/svelte/icons/arrow-down';
	import ChevronsUpDownIcon from '@lucide/svelte/icons/chevrons-up-down';

	import {
		fetch_problem_info,
		solve_scenario_rpm,
		compare_solutions,
		fetch_sessions,
		create_session
	} from './handler';

	// --- Types ---
	type ScenarioKey = 'owner_1' | 'owner_2' | 'owner_3' | 'supervisor';
	type SupervisorView = 'solve' | 'compare';
	type OwnerKey = 'owner_1' | 'owner_2' | 'owner_3';

	interface Suggestion {
		aspirationLevels: Record<string, number>;
		note: string;
	}

	interface PartyState {
		solverResults: SolverResults[] | null;
		stateId: number | null;
		preferences: Record<string, number>;
		previousPreferences: Record<string, number>;
	}

	// --- State ---
	let selection = $state<MethodSelectionState>({
		selectedProblemId: null,
		selectedMethod: null,
		selectedSessionId: null,
		selectedSessionInfo: null
	});

	let problem_info = $state<ProblemInfo | null>(null);
	let sessions = $state<InteractiveSessionBase[]>([]);
	let newSessionInfo = $state('');

	let activeTab = $state<ScenarioKey>('owner_1');
	let supervisorView = $state<SupervisorView>('solve');

	// Per-party state
	let partyStates = $state<Record<ScenarioKey, PartyState>>({
		owner_1: { solverResults: null, stateId: null, preferences: {}, previousPreferences: {} },
		owner_2: { solverResults: null, stateId: null, preferences: {}, previousPreferences: {} },
		owner_3: { solverResults: null, stateId: null, preferences: {}, previousPreferences: {} },
		supervisor: { solverResults: null, stateId: null, preferences: {}, previousPreferences: {} }
	});

	let selectedSolutionIndex = $state<Record<ScenarioKey, number>>({
		owner_1: 0,
		owner_2: 0,
		owner_3: 0,
		supervisor: 0
	});

	// Comparison
	let comparisonData = $state<ScenarioComparisonResponse | null>(null);
	let selectedClusterKeys = $state<string[]>([]);

	// Suggestion state
	let suggestions = $state<Record<OwnerKey, Suggestion | null>>({
		owner_1: null,
		owner_2: null,
		owner_3: null
	});
	let suggestionDismissed = $state<Record<OwnerKey, boolean>>({
		owner_1: false,
		owner_2: false,
		owner_3: false
	});
	let activeSuggestionCluster = $state<string | null>(null);
	let suggestionDraft = $state<Record<string, number>>({});
	let suggestionNote = $state<string>('');

	function toggleCluster(cluster: ClusterComparison) {
		const key = cluster.cluster_key;
		if (selectedClusterKeys.includes(key)) {
			selectedClusterKeys = selectedClusterKeys.filter((k) => k !== key);
		} else {
			selectedClusterKeys = [...selectedClusterKeys, key];
		}
	}

	let selectedClusters = $derived.by(() => {
		if (!comparisonData) return [];
		return comparisonData.clusters.filter((c) => selectedClusterKeys.includes(c.cluster_key));
	});

	// Derived: scenario objectives for the active tab
	let activeObjectives = $derived.by(() => {
		if (!problem_info) return [];
		return problem_info.objectives.filter((o) => {
			if (!o.scenario_keys) return false;
			return o.scenario_keys.includes(activeTab);
		});
	});

	let displayAccuracy = $derived(getDisplayAccuracy(problem_info));

	// Derived: solution rows for the current tab's table
	let currentResults = $derived(partyStates[activeTab].solverResults);

	let solutionRows = $derived.by(() => {
		if (!currentResults || !activeObjectives.length) return [];
		return currentResults.map((sr, idx) => {
			const row: Record<string, number> = {};
			for (const obj of activeObjectives) {
				row[obj.symbol] = sr.optimal_objectives[obj.symbol] ?? 0;
			}
			return row;
		});
	});

	// Derived: objective values for the visualization
	let solutionsObjectiveValues = $derived.by(() => {
		if (!currentResults || !activeObjectives.length) return [];
		return currentResults.map((sr) =>
			activeObjectives.map((obj) => sr.optimal_objectives[obj.symbol] ?? 0)
		);
	});

	// Derived: preference values as array (for visualization)
	let currentPreferenceValues = $derived.by(() => {
		const prefs = partyStates[activeTab].preferences;
		if (!activeObjectives.length || Object.keys(prefs).length === 0) return [];
		return activeObjectives.map((obj) => prefs[obj.symbol] ?? 0);
	});

	let previousPreferenceValues = $derived.by(() => {
		const prefs = partyStates[activeTab].previousPreferences;
		if (!activeObjectives.length || Object.keys(prefs).length === 0) return [];
		return [activeObjectives.map((obj) => prefs[obj.symbol] ?? 0)];
	});

	// Derived: suggestion as previousObjectiveValues for VisualizationsPanel
	let suggestionObjectiveValues = $derived.by(() => {
		if (activeTab === 'supervisor') return [];
		const suggestion = suggestions[activeTab as OwnerKey];
		if (!suggestion) return [];
		return [activeObjectives.map((obj) => suggestion.aspirationLevels[obj.symbol] ?? 0)];
	});

	let suggestionLabels = $derived.by(() => {
		if (activeTab === 'supervisor') return {};
		if (!suggestions[activeTab as OwnerKey]) return {};
		return { previousSolutionLabels: ["Supervisor's suggestion"] };
	});

	// Can compare: supervisor solved + at least one owner solved
	let canCompare = $derived.by(() => {
		if (partyStates.supervisor.stateId === null) return false;
		for (const key of ['owner_1', 'owner_2', 'owner_3'] as const) {
			if (partyStates[key].stateId !== null) return true;
		}
		return false;
	});

	// Per-cluster objectives (owner-specific symbols, used by comparison tables)
	let comparisonObjectives = $derived.by(() => {
		if (!problem_info || !comparisonData) return [];
		const keys = new Set(comparisonData.clusters.map((c) => c.cluster_key));
		return problem_info.objectives.filter((o) => {
			if (!o.scenario_keys) return false;
			return o.scenario_keys.some((k) => keys.has(k));
		});
	});

	// Canonical (owner-agnostic) dimensions for the parallel coordinates plot.
	// Objectives like NPV_owner_1, NPV_owner_2 map to a single "NPV" axis.
	// Axis ranges are the union across all owners so every value fits.
	let canonicalDimensions = $derived.by(() => {
		if (!comparisonData?.clusters.length || !problem_info) return [];
		const grouped = new Map<string, { name: string; lo: number; hi: number; maximize: boolean }>();
		for (const cluster of comparisonData.clusters) {
			for (const obj of problem_info.objectives.filter((o) =>
				o.scenario_keys?.includes(cluster.cluster_key)
			)) {
				const base = obj.symbol.replace(`_${cluster.cluster_key}`, '');
				const lo = Math.min(obj.ideal ?? 0, obj.nadir ?? 0);
				const hi = Math.max(obj.ideal ?? 0, obj.nadir ?? 0);
				const existing = grouped.get(base);
				if (!existing) {
					grouped.set(base, {
						name: obj.name.replace(/ \(.*\)$/, ''),
						lo,
						hi,
						maximize: obj.maximize
					});
				} else {
					existing.lo = Math.min(existing.lo, lo);
					existing.hi = Math.max(existing.hi, hi);
				}
			}
		}
		return Array.from(grouped, ([symbol, g]) => ({
			symbol,
			name: g.name,
			min: g.lo,
			max: g.hi,
			direction: (g.maximize ? 'max' : 'min') as 'max' | 'min'
		}));
	});

	/** Remap owner-specific keys (e.g. NPV_owner_1) to canonical keys (NPV). */
	function remapToCanonical(
		objectives: Record<string, number>,
		clusterKey: string
	): Record<string, number> {
		const result: Record<string, number> = {};
		for (const [key, value] of Object.entries(objectives)) {
			result[key.replace(`_${clusterKey}`, '')] = value;
		}
		return result;
	}

	// --- Suggestion helpers ---
	function initSuggestionDraft(cluster: ClusterComparison) {
		const draft: Record<string, number> = {};
		// Copy supervisor objectives for this cluster, keeping owner-specific keys
		for (const [key, value] of Object.entries(cluster.supervisor_objectives)) {
			draft[key] = value;
		}
		suggestionDraft = draft;
		suggestionNote = '';
		activeSuggestionCluster = cluster.cluster_key;
	}

	function sendSuggestion(ownerKey: OwnerKey) {
		suggestions[ownerKey] = {
			aspirationLevels: { ...suggestionDraft },
			note: suggestionNote
		};
		suggestionDismissed[ownerKey] = false;
		activeSuggestionCluster = null;
		suggestionDraft = {};
		suggestionNote = '';
	}

	function cancelSuggestion() {
		activeSuggestionCluster = null;
		suggestionDraft = {};
		suggestionNote = '';
	}

	function dismissSuggestion(ownerKey: OwnerKey) {
		suggestionDismissed[ownerKey] = true;
	}

	function applySuggestion(ownerKey: OwnerKey) {
		const suggestion = suggestions[ownerKey];
		if (!suggestion) return;
		for (const [key, value] of Object.entries(suggestion.aspirationLevels)) {
			partyStates[ownerKey].preferences[key] = value;
		}
	}

	// --- Table setup ---
	let tableSorting = $state<SortingState>([]);

	const tableColumns: ColumnDef<Record<string, number>>[] = $derived.by(() => {
		if (!activeObjectives.length) return [];
		return [
			{
				id: 'row_number',
				header: '#',
				cell: ({ row }: { row: any }) => String(row.index + 1),
				enableSorting: false
			},
			...activeObjectives.map((objective, idx) => ({
				accessorKey: objective.symbol,
				header: ({ column }: { column: Column<Record<string, number>> }) =>
					renderSnippet(ColumnHeader, {
						column,
						title: `${objective.name} (${objective.maximize ? 'max' : 'min'})`,
						colorIdx: idx
					}),
				cell: ({ row }: { row: any }) =>
					renderSnippet(ObjectiveCell, {
						value: row.original[objective.symbol],
						accuracy: displayAccuracy[idx] ?? 2
					}),
				enableSorting: true
			}))
		];
	});

	const table = createSvelteTable({
		get data() {
			return solutionRows;
		},
		get columns() {
			return tableColumns;
		},
		state: {
			get sorting() {
				return tableSorting;
			}
		},
		onSortingChange: (updater) => {
			if (typeof updater === 'function') {
				tableSorting = updater(tableSorting);
			} else {
				tableSorting = updater;
			}
		},
		getCoreRowModel: getCoreRowModel(),
		getSortedRowModel: getSortedRowModel()
	});

	// --- Session handling ---
	async function handleSessionSelect(value: string) {
		const id = Number(value);
		const session = sessions.find((s) => s.id === id);
		if (!session) return;
		methodSelection.setSession(id, session.info ?? null);
	}

	async function handleCreateSession() {
		const trimmed = newSessionInfo.trim();
		try {
			isLoading.set(true);
			const created = await create_session(trimmed || null);
			const fetchedSessions = await fetch_sessions();
			if (fetchedSessions) sessions = fetchedSessions;
			if (created && created.id != null) {
				methodSelection.setSession(created.id, created.info ?? null);
			}
			newSessionInfo = '';
		} catch (err) {
			console.error('Error creating session:', err);
			errorMessage.set('Failed to create session.');
		} finally {
			isLoading.set(false);
		}
	}

	// --- Solve ---
	async function handleSolve() {
		if (selection.selectedProblemId === null) {
			errorMessage.set('No problem selected.');
			return;
		}

		const prefs = partyStates[activeTab].preferences;
		if (Object.keys(prefs).length === 0) {
			errorMessage.set('Please set aspiration levels for all objectives.');
			return;
		}

		try {
			isLoading.set(true);

			// Save current preferences as previous before solving
			const prevPrefs = { ...prefs };

			const result = await solve_scenario_rpm({
				problem_id: selection.selectedProblemId,
				session_id: selection.selectedSessionId ?? undefined,
				scenario_key: activeTab,
				preference: { aspiration_levels: prefs }
			});

			if (!result) {
				errorMessage.set('Solve failed.');
				return;
			}

			partyStates[activeTab] = {
				...partyStates[activeTab],
				solverResults: result.solver_results,
				stateId: result.id,
				previousPreferences: prevPrefs
			};
			selectedSolutionIndex[activeTab] = 0;
		} catch (err) {
			console.error('Solve error:', err);
			errorMessage.set('Unexpected error during solve.');
		} finally {
			isLoading.set(false);
		}
	}

	// --- Compare ---
	async function handleCompare() {
		if (selection.selectedProblemId === null || partyStates.supervisor.stateId === null) {
			return;
		}

		// Build owner_states dict from all solved owners
		const ownerStates: Record<string, number> = {};
		for (const key of ['owner_1', 'owner_2', 'owner_3'] as const) {
			const stateId = partyStates[key].stateId;
			if (stateId !== null) {
				ownerStates[key] = stateId;
			}
		}

		if (Object.keys(ownerStates).length === 0) {
			errorMessage.set('No owners have solved yet.');
			return;
		}

		try {
			isLoading.set(true);
			const result = await compare_solutions({
				problem_id: selection.selectedProblemId,
				supervisor_state_id: partyStates.supervisor.stateId!,
				owner_states: ownerStates
			});

			if (!result) {
				errorMessage.set('Comparison failed.');
				return;
			}

			comparisonData = result;
		} catch (err) {
			console.error('Compare error:', err);
			errorMessage.set('Unexpected error during comparison.');
		} finally {
			isLoading.set(false);
		}
	}

	// --- Lifecycle ---
	onMount(() => {
		const unsubscribe = methodSelection.subscribe((v) => (selection = v));

		(async () => {
			if (selection.selectedProblemId === null) {
				console.log('No problem selected for ScenarioRPM.');
				return;
			}
			try {
				isLoading.set(true);

				const request: ProblemGetRequest = { problem_id: selection.selectedProblemId };
				const response = await fetch_problem_info(request);
				if (!response) {
					errorMessage.set('Could not fetch problem information.');
					return;
				}
				problem_info = response;

				// Initialize preferences at midpoint of ideal/nadir for each scenario
				for (const key of ['owner_1', 'owner_2', 'owner_3', 'supervisor'] as ScenarioKey[]) {
					const objectives = problem_info.objectives.filter(
						(o) => o.scenario_keys?.includes(key)
					);
					const prefs: Record<string, number> = {};
					for (const obj of objectives) {
						const ideal = obj.ideal ?? 0;
						const nadir = obj.nadir ?? 0;
						prefs[obj.symbol] = (ideal + nadir) / 2;
					}
					partyStates[key] = { ...partyStates[key], preferences: prefs };
				}

				const fetchedSessions = await fetch_sessions();
				sessions = fetchedSessions ?? [];
			} catch (err) {
				console.error('Error during ScenarioRPM init:', err);
				errorMessage.set('Unexpected error during initialization.');
			} finally {
				isLoading.set(false);
			}
		})();

		return unsubscribe;
	});

	// Helper: tab display name
	function tabLabel(key: ScenarioKey): string {
		if (key === 'supervisor') return 'Supervisor';
		return key.replace('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase());
	}
</script>

{#snippet ColumnHeader({ column, title, colorIdx }: { column: Column<Record<string, number>>; title: string; colorIdx?: number })}
	<div
		class="flex items-center"
		style={colorIdx != null
			? `border-bottom: 6px solid ${COLOR_PALETTE[colorIdx % COLOR_PALETTE.length]}; width: 100%; padding: 0.5rem;`
			: ''}
	>
		{#if column.getCanSort()}
			<Button
				variant="ghost"
				size="sm"
				onclick={() => column.toggleSorting(column.getIsSorted() === 'asc')}
				class="-ml-3 h-8"
			>
				<span>{title}</span>
				{#if column.getIsSorted() === 'desc'}
					<ArrowDownIcon class="ml-2 h-4 w-4" />
				{:else if column.getIsSorted() === 'asc'}
					<ArrowUpIcon class="ml-2 h-4 w-4" />
				{:else}
					<ChevronsUpDownIcon class="ml-2 h-4 w-4 opacity-50" />
				{/if}
			</Button>
		{:else}
			<span class="px-2 py-1">{title}</span>
		{/if}
	</div>
{/snippet}

{#snippet ObjectiveCell({ value, accuracy }: { value: number | null | undefined; accuracy: number })}
	<div class="pr-4 text-right">
		{value != null ? formatNumber(value, accuracy) : '-'}
	</div>
{/snippet}

<!-- Session selector (shown when no session is selected) -->
{#if !selection.selectedSessionId}
	<div class="mx-auto max-w-md space-y-4 p-8">
		<h2 class="text-lg font-semibold">Select or Create a Session</h2>

		{#if sessions.length > 0}
			<Combobox
				options={sessions.map((s) => ({ value: String(s.id), label: s.info ?? `Session ${s.id}` }))}
				defaultSelected=""
				onChange={(e) => handleSessionSelect(e.value)}
				placeholder="Select a session..."
			/>
		{/if}

		<div class="flex gap-2">
			<Input bind:value={newSessionInfo} placeholder="Session name..." class="flex-1" />
			<Button onclick={handleCreateSession}>Create</Button>
		</div>
	</div>
{:else}
	<!-- Main method interface -->
	<div class="flex flex-col" style="height: calc(100vh - 3rem);">
		<!-- Top tab bar -->
		<div class="flex-shrink-0 border-b px-4 pt-2">
			<Tabs.Root bind:value={activeTab}>
				<Tabs.List>
					{#each ['owner_1', 'owner_2', 'owner_3', 'supervisor'] as key}
						<Tabs.Trigger value={key}>
							{tabLabel(key as ScenarioKey)}
							{#if partyStates[key as ScenarioKey].stateId !== null}
								<span class="ml-1 inline-block h-2 w-2 rounded-full bg-green-500"></span>
							{/if}
						</Tabs.Trigger>
					{/each}
				</Tabs.List>
			</Tabs.Root>

			<!-- Supervisor sub-tabs -->
			{#if activeTab === 'supervisor'}
				<div class="mt-1 flex gap-2">
					<Button
						variant={supervisorView === 'solve' ? 'default' : 'outline'}
						size="sm"
						onclick={() => (supervisorView = 'solve')}
					>
						Solve
					</Button>
					<Button
						variant={supervisorView === 'compare' ? 'default' : 'outline'}
						size="sm"
						onclick={() => (supervisorView = 'compare')}
					>
						Compare
					</Button>
				</div>
			{/if}
		</div>

		<!-- Content -->
		{#if activeTab === 'supervisor' && supervisorView === 'compare'}
			<!-- Comparison view -->
			<div class="flex min-h-0 flex-1 flex-col p-4">
				<div class="mb-4 flex items-center gap-4">
					<Button onclick={handleCompare} disabled={!canCompare}>
						Run Comparison
					</Button>
					{#if !canCompare}
						<p class="text-sm text-gray-400">
							Supervisor and at least one owner must solve first.
						</p>
					{/if}
				</div>

				<div class="flex min-h-0 flex-1 gap-4">
					<!-- Left: honeycomb -->
					<div class="flex min-h-0 flex-1 flex-col">
						<div class="relative min-h-0 flex-1 rounded border bg-gray-50 p-4">
							<Honeycomb
								data={comparisonData}
								onclusterclick={toggleCluster}
								selectedClusterKeys={selectedClusterKeys}
							/>
						</div>

						{#if comparisonData}
							<div class="mt-2 flex items-center gap-3 text-xs text-gray-500">
								<span>Distance:</span>
								<span>Identical</span>
								<div
									class="h-3 w-32 rounded"
									style="background: {DISTANCE_GRADIENT_CSS};"
								></div>
								<span>Max difference</span>
							</div>
						{/if}
					</div>

					<!-- Right: detail panel (always mounted, data updates in place) -->
					<div class="flex w-96 shrink-0 flex-col rounded-lg border border-gray-200 bg-white shadow-sm">
						<div class="flex items-center justify-between border-b border-gray-100 px-4 py-3">
							<span class="text-sm font-semibold text-gray-800">
								{#if selectedClusters.length > 0}
									{selectedClusters.map((c) => c.cluster_key.replace('_', ' ').replace(/\b\w/g, (ch) => ch.toUpperCase())).join(', ')}
								{:else}
									Cluster Details
								{/if}
							</span>
							{#if selectedClusters.length > 0}
								<button
									class="text-gray-400 hover:text-gray-600"
									onclick={() => (selectedClusterKeys = [])}
								>
									&times;
								</button>
							{/if}
						</div>
						<div class="flex-1 overflow-auto">
							<!-- Parallel coordinates plot (always rendered) -->
							<div class="h-56 px-2 pt-2">
								<ParallelCoordinates
									data={selectedClusters.map((c) => remapToCanonical(c.owner_objectives, c.cluster_key))}
									dimensions={canonicalDimensions}
									referenceData={selectedClusters.length > 0
										? {
												referencePoint: {
													values: remapToCanonical(selectedClusters[0].supervisor_objectives, selectedClusters[0].cluster_key),
													label: 'Supervisor'
												}
											}
										: undefined}
									lineLabels={Object.fromEntries(
										selectedClusters.map((c, i) => [
											i,
											c.cluster_key.replace('_', ' ').replace(/\b\w/g, (ch) => ch.toUpperCase())
										])
									)}
									lineColors={CLUSTER_BORDER_COLORS}
									options={{
										showAxisLabels: true,
										highlightOnHover: true,
										strokeWidth: 2,
										opacity: 0.9,
										enableBrushing: false
									}}
								/>
							</div>
							<!-- Legend -->
							<div class="flex flex-wrap items-center justify-center gap-x-4 gap-y-1 px-4 pb-1 text-xs text-gray-600">
								{#each selectedClusters as cluster, i}
									<span class="flex items-center gap-1.5">
										<span class="inline-block h-0.5 w-4" style="background-color: {CLUSTER_BORDER_COLORS[i % CLUSTER_BORDER_COLORS.length]};"></span>
										{cluster.cluster_key.replace('_', ' ').replace(/\b\w/g, (ch) => ch.toUpperCase())}
									</span>
								{/each}
								{#if selectedClusters.length > 0}
									<span class="flex items-center gap-1.5">
										<span class="inline-block h-0.5 w-4 border-t-2 border-dashed border-red-400"></span>
										Supervisor
									</span>
								{/if}
							</div>

							{#if selectedClusters.length === 0}
								<div class="flex flex-1 items-center justify-center py-8">
									<p class="text-sm text-gray-400">Click a cluster to see details.</p>
								</div>
							{:else}
								<!-- Numerical comparison tables (one per cluster) -->
								{#each selectedClusters as cluster, ci}
									{@const clusterLabel = cluster.cluster_key.replace('_', ' ').replace(/\b\w/g, (ch) => ch.toUpperCase())}
									<div class="px-4 py-3" class:border-t={ci > 0} class:border-gray-100={ci > 0}>
										{#if selectedClusters.length > 1}
											<p class="mb-1 text-xs font-semibold text-gray-700">{clusterLabel}</p>
										{/if}
										<table class="w-full text-xs">
											<thead>
												<tr class="border-b border-gray-200">
													<th class="pb-1 pr-2 text-left font-medium text-gray-500">Objective</th>
													<th class="pb-1 pr-2 text-right font-medium text-gray-500">Supervisor</th>
													<th class="pb-1 pr-2 text-right font-medium text-gray-500">Owner</th>
													<th class="pb-1 text-right font-medium text-gray-500">Delta</th>
												</tr>
											</thead>
											<tbody>
												{#each comparisonObjectives.filter((o) => o.scenario_keys?.includes(cluster.cluster_key)) as obj, idx}
													{@const sv = cluster.supervisor_objectives[obj.symbol]}
													{@const ov = cluster.owner_objectives[obj.symbol]}
													{@const delta = sv - ov}
													{@const acc = displayAccuracy[idx] ?? 2}
													<tr class="border-b border-gray-50">
														<td class="py-1 pr-2 text-gray-700">{obj.name}</td>
														<td class="py-1 pr-2 text-right">{formatNumber(sv, acc)}</td>
														<td class="py-1 pr-2 text-right">{formatNumber(ov, acc)}</td>
														<td
															class="py-1 text-right"
															class:text-red-600={delta < 0}
															class:text-green-600={delta > 0}
														>
															{delta >= 0 ? '+' : ''}{formatNumber(delta, acc)}
														</td>
													</tr>
												{/each}
											</tbody>
										</table>
										<div class="mt-1 text-xs text-gray-500">
											Distance: {formatNumber(cluster.distance, 3)}
										</div>

										<!-- Suggestion controls -->
										{#if activeSuggestionCluster === cluster.cluster_key}
											<div class="mt-3 rounded-md border-2 border-emerald-300 bg-emerald-50 p-3">
												<p class="mb-2 text-xs font-semibold text-emerald-800">Suggest aspiration levels for {clusterLabel}</p>
												{#each comparisonObjectives.filter((o) => o.scenario_keys?.includes(cluster.cluster_key)) as obj, idx}
													{@const ideal = obj.ideal ?? 0}
													{@const nadir = obj.nadir ?? 0}
													{@const lo = Math.min(ideal, nadir)}
													{@const hi = Math.max(ideal, nadir)}
													<div class="mb-1">
														<Label class="text-xs">
															{obj.name}
															<span class="text-gray-400">({obj.maximize ? 'max' : 'min'})</span>
														</Label>
														<HorizontalBar
															axisRanges={[lo, hi]}
															selectedValue={suggestionDraft[obj.symbol] ?? (lo + hi) / 2}
															barColor="#10b981"
															direction={obj.maximize ? 'max' : 'min'}
															options={{
																decimalPrecision: displayAccuracy[idx] ?? 2,
																showSelectedValueLabel: false,
																aspectRatio: 'aspect-[11/2]'
															}}
															onSelect={(v) => {
																suggestionDraft[obj.symbol] = v;
															}}
														/>
													</div>
												{/each}
												<div class="mt-2">
													<Label class="text-xs">Note for the owner</Label>
													<Textarea
														bind:value={suggestionNote}
														placeholder="Write a note to the owner..."
														class="mt-1 text-xs"
														rows={2}
													/>
												</div>
												<div class="mt-2 flex gap-2">
													<Button size="sm" class="bg-emerald-600 hover:bg-emerald-700" onclick={() => sendSuggestion(cluster.cluster_key as OwnerKey)}>Send</Button>
													<Button size="sm" variant="outline" onclick={cancelSuggestion}>Cancel</Button>
												</div>
											</div>
										{:else}
											<div class="mt-2 flex items-center gap-2">
												{#if suggestions[cluster.cluster_key as OwnerKey]}
													<span class="text-xs text-emerald-600">Suggestion sent</span>
													<Button size="sm" variant="ghost" class="h-6 text-xs text-emerald-600" onclick={() => initSuggestionDraft(cluster)}>Edit</Button>
												{:else}
													<Button size="sm" variant="outline" class="h-7 text-xs" onclick={() => initSuggestionDraft(cluster)}>Make Suggestion</Button>
												{/if}
											</div>
										{/if}
									</div>
								{/each}
							{/if}
						</div>
					</div>
				</div>
			</div>
		{:else}
			<!-- Solve view (shared for owners and supervisor-solve) -->
			<BaseLayout showRightSidebar={false}>
				{#snippet leftSidebar()}
					<div class="w-64 space-y-4 p-4">
						<h3 class="text-sm font-semibold">Reference Point</h3>
						<p class="text-xs text-gray-500">{tabLabel(activeTab)}</p>

						<!-- Supervisor suggestion banner -->
						{#if activeTab !== 'supervisor' && suggestions[activeTab as OwnerKey] && !suggestionDismissed[activeTab as OwnerKey]}
							<div class="rounded-md border-2 border-emerald-300 bg-emerald-50 p-3">
								<div class="flex items-start justify-between">
									<p class="text-xs font-semibold text-emerald-800">Supervisor's Suggestion</p>
									<button
										class="text-emerald-400 hover:text-emerald-600"
										onclick={() => dismissSuggestion(activeTab as OwnerKey)}
									>
										&times;
									</button>
								</div>
								{#if suggestions[activeTab as OwnerKey]?.note}
									<p class="mt-1 text-xs italic text-emerald-700">{suggestions[activeTab as OwnerKey]?.note}</p>
								{/if}
								<Button
									size="sm"
									variant="outline"
									class="mt-2 h-7 border-emerald-400 text-xs text-emerald-700 hover:bg-emerald-100"
									onclick={() => applySuggestion(activeTab as OwnerKey)}
								>
									Apply to sliders
								</Button>
							</div>
						{/if}

						{#each activeObjectives as obj, idx}
							{@const ideal = obj.ideal ?? 0}
							{@const nadir = obj.nadir ?? 0}
							{@const lo = Math.min(ideal, nadir)}
							{@const hi = Math.max(ideal, nadir)}
							{@const solIdx = selectedSolutionIndex[activeTab]}
							{@const solVal = currentResults?.[solIdx]?.optimal_objectives?.[obj.symbol]}
							{@const prevVal = partyStates[activeTab].previousPreferences[obj.symbol]}
							<div>
								<Label class="text-xs">
									{obj.name}
									<span class="text-gray-400">({obj.maximize ? 'max' : 'min'})</span>
								</Label>
								<HorizontalBar
									axisRanges={[lo, hi]}
									solutionValue={solVal}
									selectedValue={partyStates[activeTab].preferences[obj.symbol] ?? (lo + hi) / 2}
									previousValue={prevVal}
									barColor={COLOR_PALETTE[idx % COLOR_PALETTE.length]}
									direction={obj.maximize ? 'max' : 'min'}
									options={{
										decimalPrecision: displayAccuracy[idx] ?? 2,
										showPreviousValue: prevVal !== undefined,
										showSelectedValueLabel: false,
										aspectRatio: 'aspect-[11/2]'
									}}
									onSelect={(v) => {
										partyStates[activeTab].preferences[obj.symbol] = v;
									}}
								/>
							</div>
						{/each}

						<Button onclick={handleSolve} class="w-full">Solve</Button>
					</div>
				{/snippet}

				{#snippet explorerTitle()}
					{tabLabel(activeTab)} — Solution Explorer
				{/snippet}

				{#snippet visualizationArea()}
					{#if problem_info && solutionsObjectiveValues.length > 0}
						<VisualizationsPanel
							problem={{
								...problem_info,
								objectives: activeObjectives
							}}
							previousPreferenceValues={previousPreferenceValues}
							currentPreferenceValues={currentPreferenceValues}
							previousPreferenceType={previousPreferenceValues.length > 0 ? 'reference_point' : ''}
							currentPreferenceType="reference_point"
							solutionsObjectiveValues={solutionsObjectiveValues}
							previousObjectiveValues={suggestionObjectiveValues}
							referenceDataLabels={suggestionLabels}
							externalSelectedIndexes={[selectedSolutionIndex[activeTab]]}
						/>
					{:else}
						<div class="flex h-full items-center justify-center text-sm text-gray-400">
							Set aspiration levels and click Solve to generate solutions.
						</div>
					{/if}
				{/snippet}

				{#snippet numericalValues()}
					{#if solutionRows.length > 0}
						<div class="max-h-64 overflow-auto">
							<Table.Root>
								<Table.Header>
									{#each table.getHeaderGroups() as headerGroup}
										<Table.Row>
											{#each headerGroup.headers as header}
												<Table.Head>
													{#if !header.isPlaceholder}
														<FlexRender
															content={header.column.columnDef.header}
															context={header.getContext()}
														/>
													{/if}
												</Table.Head>
											{/each}
										</Table.Row>
									{/each}
								</Table.Header>
								<Table.Body>
									{#each table.getRowModel().rows as row}
										<Table.Row
											class={selectedSolutionIndex[activeTab] === row.index
												? 'bg-blue-50'
												: 'cursor-pointer hover:bg-gray-50'}
											onclick={() => (selectedSolutionIndex[activeTab] = row.index)}
										>
											{#each row.getVisibleCells() as cell}
												<Table.Cell>
													<FlexRender
														content={cell.column.columnDef.cell}
														context={cell.getContext()}
													/>
												</Table.Cell>
											{/each}
										</Table.Row>
									{/each}
								</Table.Body>
							</Table.Root>
						</div>
					{:else}
						<div class="p-4 text-sm text-gray-400">No solutions yet.</div>
					{/if}
				{/snippet}
			</BaseLayout>
		{/if}
	</div>
{/if}
