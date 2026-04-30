<script lang="ts">
	import type { SiteInfo, SolverResults } from '$lib/gen/models';

	type ConstraintState = 'free' | 'restricted' | 'forced';

	interface Props {
		sites: SiteInfo[];
		baseSolution: SolverResults | null;
		whatIfSolution: SolverResults | null;
		baseLabel: string;
		whatIfLabel: string;
		siteStates: Map<string, ConstraintState>;
	}

	let {
		sites,
		baseSolution,
		whatIfSolution,
		baseLabel,
		whatIfLabel,
		siteStates = $bindable(),
	}: Props = $props();

	let search = $state('');
	let cityFilter = $state<string>('');
	let stateFilter = $state<'all' | 'free' | 'forced' | 'restricted'>('all');

	let allCities = $derived.by(() => {
		const set = new Set<string>();
		for (const s of sites) set.add(s.node);
		return Array.from(set).sort();
	});

	function visitedSymbols(sol: SolverResults | null): Set<string> {
		const visited = new Set<string>();
		if (!sol) return visited;
		const vars = sol.optimal_variables as Record<string, unknown> | undefined;
		if (!vars) return visited;
		for (const s of sites) {
			const v = vars[s.variable_symbol];
			if (Array.isArray(v)) {
				if (Math.round(Number(v[0])) === 1) visited.add(s.variable_symbol);
			} else if (v != null && Math.round(Number(v)) === 1) {
				visited.add(s.variable_symbol);
			}
		}
		const sv = vars['sv'];
		if (Array.isArray(sv)) {
			let idx = 1;
			const flatten = (arr: unknown[]): void => {
				for (const el of arr) {
					if (Array.isArray(el)) flatten(el);
					else {
						if (Math.round(Number(el)) === 1) visited.add(`sv_${idx}`);
						idx++;
					}
				}
			};
			flatten(sv);
		}
		return visited;
	}

	let baseVisited = $derived(visitedSymbols(baseSolution));
	let whatIfVisited = $derived(visitedSymbols(whatIfSolution));

	let visibleSites = $derived.by(() => {
		const q = search.trim().toLowerCase();
		return sites
			.filter((s) => {
				if (cityFilter && s.node !== cityFilter) return false;
				const state = siteStates.get(s.variable_symbol) ?? 'free';
				if (stateFilter !== 'all' && state !== stateFilter) return false;
				if (q) {
					if (
						!s.name.toLowerCase().includes(q) &&
						!s.node.toLowerCase().includes(q) &&
						!s.variable_symbol.toLowerCase().includes(q)
					) {
						return false;
					}
				}
				return true;
			})
			.sort((a, b) => a.node.localeCompare(b.node) || a.name.localeCompare(b.name));
	});

	function cycleSite(symbol: string) {
		const cur = siteStates.get(symbol) ?? 'free';
		const next: ConstraintState = cur === 'free' ? 'restricted' : cur === 'restricted' ? 'forced' : 'free';
		const nextMap = new Map(siteStates);
		if (next === 'free') nextMap.delete(symbol);
		else nextMap.set(symbol, next);
		siteStates = nextMap;
	}

	function setSite(symbol: string, target: ConstraintState | 'free') {
		const nextMap = new Map(siteStates);
		if (target === 'free') nextMap.delete(symbol);
		else nextMap.set(symbol, target);
		siteStates = nextMap;
	}

	function clearAll() {
		siteStates = new Map();
	}

	function setBulk(target: ConstraintState | 'free') {
		const nextMap = new Map(siteStates);
		for (const s of visibleSites) {
			if (target === 'free') nextMap.delete(s.variable_symbol);
			else nextMap.set(s.variable_symbol, target);
		}
		siteStates = nextMap;
	}

	function stateLabel(s: ConstraintState): string {
		if (s === 'forced') return 'Forced';
		if (s === 'restricted') return 'Excluded';
		return 'Free';
	}

	function stateClass(s: ConstraintState): string {
		if (s === 'forced') return 'bg-green-100 text-green-800 border-green-300';
		if (s === 'restricted') return 'bg-red-100 text-red-800 border-red-300';
		return 'bg-gray-100 text-gray-700 border-gray-300';
	}
</script>

<div>
	<div class="flex flex-wrap items-center gap-2 border-b bg-gray-50 px-3 py-2 text-xs">
		<input
			type="text"
			bind:value={search}
			placeholder="Search by site, city or symbol..."
			class="w-48 rounded border border-gray-300 px-2 py-1"
		/>
		<div
			class="flex items-center gap-2 rounded border border-gray-200 bg-white px-2 py-1 text-[11px] text-gray-600"
			title="Visit status indicators used in the table cells below."
		>
			<span class="font-semibold text-gray-700">Legend:</span>
			<span class="inline-flex items-center gap-1" title="Site is visited in this solution">
				<span class="inline-block h-2.5 w-2.5 rounded-full bg-orange-400"></span>visited
			</span>
			<span class="inline-flex items-center gap-1" title="Site is not visited in this solution">
				<span class="inline-block h-2.5 w-2.5 rounded-full border border-gray-300 bg-white"></span>not&nbsp;visited
			</span>
			{#if whatIfSolution}
				<span class="inline-flex items-center gap-1" title="What-if adds this site (was not visited in the baseline)">
					<span class="inline-block h-2.5 w-2.5 rounded-full bg-emerald-500"></span>added
				</span>
				<span class="inline-flex items-center gap-1" title="What-if removes this site (was visited in the baseline)">
					<span class="inline-block h-2.5 w-2.5 rounded-full bg-amber-500"></span>removed
				</span>
			{/if}
		</div>
		<select bind:value={cityFilter} class="rounded border border-gray-300 px-2 py-1">
			<option value="">All cities</option>
			{#each allCities as c (c)}
				<option value={c}>{c}</option>
			{/each}
		</select>
		<select bind:value={stateFilter} class="rounded border border-gray-300 px-2 py-1">
			<option value="all">All states</option>
			<option value="free">Free</option>
			<option value="forced">Forced</option>
			<option value="restricted">Excluded</option>
		</select>
		<span class="text-gray-500">{visibleSites.length} of {sites.length}</span>
		<div class="ml-auto flex items-center gap-1">
			<span class="text-gray-500">Apply to filtered:</span>
			<button
				class="rounded border border-green-300 bg-green-50 px-2 py-1 text-green-700 hover:bg-green-100"
				onclick={() => setBulk('forced')}
			>Force</button>
			<button
				class="rounded border border-red-300 bg-red-50 px-2 py-1 text-red-700 hover:bg-red-100"
				onclick={() => setBulk('restricted')}
			>Exclude</button>
			<button
				class="rounded border border-gray-300 bg-gray-50 px-2 py-1 text-gray-700 hover:bg-gray-100"
				onclick={() => setBulk('free')}
			>Clear</button>
			<button
				class="ml-2 rounded border border-gray-300 bg-white px-2 py-1 text-gray-700 hover:bg-gray-100"
				onclick={clearAll}
				title="Clear constraints on every site"
			>Clear all</button>
		</div>
	</div>

	<table class="w-full border-collapse text-sm">
			<thead class="sticky top-0 z-10 bg-white shadow-[0_1px_0_rgb(229_231_235)]">
				<tr>
					<th class="px-3 py-2 text-left font-semibold">City</th>
					<th class="px-3 py-2 text-left font-semibold">Site</th>
					<th class="px-3 py-2 text-left font-semibold text-gray-500">Symbol</th>
					<th
						class="px-3 py-2 text-center font-semibold"
						title="Whether this site is visited in the {baseLabel.toLowerCase()} solution. Orange = visited, empty = not visited."
					>{baseLabel}</th>
					{#if whatIfSolution}
						<th
							class="px-3 py-2 text-center font-semibold"
							title="Whether this site is visited in the what-if solution. Green = newly added, amber = removed compared to {baseLabel.toLowerCase()}, orange = visited in both, empty = visited in neither."
						>{whatIfLabel}</th>
					{/if}
					<th class="px-3 py-2 text-center font-semibold">State</th>
					<th class="px-3 py-2 text-center font-semibold">Action</th>
				</tr>
			</thead>
			<tbody>
				{#each visibleSites as site (site.variable_symbol)}
					{@const state = siteStates.get(site.variable_symbol) ?? 'free'}
					{@const inBase = baseVisited.has(site.variable_symbol)}
					{@const inWhatIf = whatIfVisited.has(site.variable_symbol)}
					<tr class="border-b border-gray-100 hover:bg-gray-50">
						<td class="px-3 py-1.5 text-gray-700">{site.node}</td>
						<td class="px-3 py-1.5 font-medium">{site.name}</td>
						<td class="px-3 py-1.5 font-mono text-[11px] text-gray-500">{site.variable_symbol}</td>
						<td class="px-3 py-1.5 text-center">
							{#if inBase}
								<span class="inline-block h-2.5 w-2.5 rounded-full bg-orange-400" title="Visited"></span>
							{:else}
								<span class="inline-block h-2.5 w-2.5 rounded-full border border-gray-300 bg-white" title="Not visited"></span>
							{/if}
						</td>
						{#if whatIfSolution}
							<td class="px-3 py-1.5 text-center">
								{#if inWhatIf && !inBase}
									<span class="inline-block h-2.5 w-2.5 rounded-full bg-emerald-500" title="Added in what-if"></span>
								{:else if inBase && !inWhatIf}
									<span class="inline-block h-2.5 w-2.5 rounded-full bg-amber-500" title="Removed in what-if"></span>
								{:else if inWhatIf}
									<span class="inline-block h-2.5 w-2.5 rounded-full bg-orange-400" title="Visited"></span>
								{:else}
									<span class="inline-block h-2.5 w-2.5 rounded-full border border-gray-300 bg-white" title="Not visited"></span>
								{/if}
							</td>
						{/if}
						<td class="px-3 py-1.5 text-center">
							<span class={`rounded border px-2 py-0.5 text-[11px] ${stateClass(state)}`}>
								{stateLabel(state)}
							</span>
						</td>
						<td class="px-3 py-1.5 text-center">
							<div class="inline-flex gap-1">
								<button
									class="rounded border border-green-300 bg-green-50 px-2 py-0.5 text-[11px] text-green-700 hover:bg-green-100 disabled:opacity-30"
									disabled={state === 'forced'}
									onclick={() => setSite(site.variable_symbol, 'forced')}
									title="Force this site to be visited"
								>Force</button>
								<button
									class="rounded border border-red-300 bg-red-50 px-2 py-0.5 text-[11px] text-red-700 hover:bg-red-100 disabled:opacity-30"
									disabled={state === 'restricted'}
									onclick={() => setSite(site.variable_symbol, 'restricted')}
									title="Exclude this site"
								>Exclude</button>
								<button
									class="rounded border border-gray-300 bg-gray-50 px-2 py-0.5 text-[11px] text-gray-700 hover:bg-gray-100 disabled:opacity-30"
									disabled={state === 'free'}
									onclick={() => setSite(site.variable_symbol, 'free')}
									title="Remove this constraint"
								>Free</button>
								<button
									class="rounded border border-gray-300 bg-white px-2 py-0.5 text-[11px] text-gray-500 hover:bg-gray-100"
									onclick={() => cycleSite(site.variable_symbol)}
									title="Cycle: free → exclude → force → free"
								>↻</button>
							</div>
						</td>
					</tr>
				{:else}
					<tr>
						<td colspan="7" class="px-3 py-6 text-center text-sm text-gray-400">No sites match the filters.</td>
					</tr>
				{/each}
			</tbody>
		</table>
</div>
