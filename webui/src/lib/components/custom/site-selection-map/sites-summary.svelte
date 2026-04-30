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
		siteStates,
	}: Props = $props();

	function visitedSymbols(sol: SolverResults | null): Set<string> {
		const visited = new Set<string>();
		if (!sol) return visited;
		const vars = sol.optimal_variables as Record<string, unknown> | undefined;
		if (!vars) return visited;
		// Two possible shapes: unrolled `sv_i` keys, or tensor `sv: [[v], ...]`.
		for (const s of sites) {
			const v = vars[s.variable_symbol];
			if (Array.isArray(v)) {
				if (Math.round(Number(v[0])) === 1) visited.add(s.variable_symbol);
			} else if (v != null && Math.round(Number(v)) === 1) {
				visited.add(s.variable_symbol);
			}
		}
		// Also handle tensor form.
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

	let perCity = $derived.by(() => {
		const groups = new Map<string, SiteInfo[]>();
		for (const s of sites) {
			const list = groups.get(s.node);
			if (list) list.push(s);
			else groups.set(s.node, [s]);
		}
		const rows: Array<{
			city: string;
			total: number;
			baseSites: SiteInfo[];
			whatIfSites: SiteInfo[];
			added: SiteInfo[];
			removed: SiteInfo[];
		}> = [];
		for (const [city, list] of groups) {
			const baseSites = list.filter((s) => baseVisited.has(s.variable_symbol));
			const whatIfSites = whatIfSolution
				? list.filter((s) => whatIfVisited.has(s.variable_symbol))
				: [];
			const baseSet = new Set(baseSites.map((s) => s.variable_symbol));
			const whatIfSet = new Set(whatIfSites.map((s) => s.variable_symbol));
			const added = whatIfSolution ? whatIfSites.filter((s) => !baseSet.has(s.variable_symbol)) : [];
			const removed = whatIfSolution ? baseSites.filter((s) => !whatIfSet.has(s.variable_symbol)) : [];
			rows.push({ city, total: list.length, baseSites, whatIfSites, added, removed });
		}
		rows.sort((a, b) => a.city.localeCompare(b.city));
		return rows;
	});

	let totals = $derived({
		base: baseVisited.size,
		whatIf: whatIfVisited.size,
		hasWhatIf: whatIfSolution != null,
	});

	let totalConstrained = $derived.by(() => {
		let forced = 0;
		let excluded = 0;
		for (const [, s] of siteStates) {
			if (s === 'forced') forced++;
			else if (s === 'restricted') excluded++;
		}
		return { forced, excluded };
	});

	function siteName(s: SiteInfo) {
		return `${s.name}`;
	}
</script>

<div>
	<div class="flex flex-wrap items-center gap-3 border-b bg-gray-50 px-3 py-2 text-xs">
		<div>
			<span class="text-gray-500">Total sites visited in {baseLabel}:</span>
			<span class="font-semibold text-gray-900">{totals.base}</span>
		</div>
		{#if totals.hasWhatIf}
			<div>
				<span class="text-gray-500">Total sites visited in {whatIfLabel}:</span>
				<span class="font-semibold text-gray-900">{totals.whatIf}</span>
				{#if totals.whatIf !== totals.base}
					{@const d = totals.whatIf - totals.base}
					<span class="ml-1 {d > 0 ? 'text-emerald-700' : 'text-amber-700'}">
						({d > 0 ? '+' : ''}{d})
					</span>
				{/if}
			</div>
		{/if}
		{#if totals.hasWhatIf}
			<div
				class="ml-auto flex items-center gap-2 rounded border border-gray-200 bg-white px-2 py-1 text-[11px] text-gray-600"
				title="How site visits change between the baseline and the what-if solution."
			>
				<span class="font-semibold text-gray-700">Legend:</span>
				<span
					class="rounded border border-emerald-400 bg-emerald-50 px-1.5 py-0.5 text-emerald-800"
					title="Site newly visited in the what-if (was not visited in the baseline)"
				>+ added</span>
				<span
					class="rounded border border-amber-400 bg-amber-50 px-1.5 py-0.5 text-amber-800 line-through"
					title="Site removed in the what-if (was visited in the baseline)"
				>− removed</span>
				<span
					class="rounded border border-gray-300 bg-gray-50 px-1.5 py-0.5"
					title="Site visited in both baseline and what-if"
				>unchanged</span>
			</div>
		{/if}
		{#if totalConstrained.forced > 0 || totalConstrained.excluded > 0}
			<div class="text-gray-500">
				Constraints:
				{#if totalConstrained.forced > 0}
					<span class="text-green-700 font-semibold">{totalConstrained.forced} forced</span>
				{/if}
				{#if totalConstrained.forced > 0 && totalConstrained.excluded > 0}<span>, </span>{/if}
				{#if totalConstrained.excluded > 0}
					<span class="text-red-700 font-semibold">{totalConstrained.excluded} excluded</span>
				{/if}
			</div>
		{/if}
	</div>

	<table class="w-full border-collapse text-sm">
			<thead class="sticky top-0 z-10 bg-white shadow-[0_1px_0_rgb(229_231_235)]">
				<tr>
					<th class="px-3 py-2 text-left font-semibold">City</th>
					<th
						class="px-3 py-2 text-right font-semibold"
						title="Total number of candidate sites in this city"
					>Available</th>
					<th
						class="px-3 py-2 text-right font-semibold"
						title="Number of sites visited in the {baseLabel.toLowerCase()} solution"
					>{baseLabel}</th>
					{#if totals.hasWhatIf}
						<th
							class="px-3 py-2 text-right font-semibold"
							title="Number of sites visited in the what-if solution"
						>{whatIfLabel}</th>
						<th
							class="px-3 py-2 text-right font-semibold"
							title="Change in visited sites: what-if minus {baseLabel.toLowerCase()}"
						>Δ</th>
					{/if}
					<th
						class="px-3 py-2 text-left font-semibold"
						title="Sites visited in this city. With a what-if active, green chips are added by the what-if and amber strike-through chips are removed."
					>Visited sites</th>
				</tr>
			</thead>
			<tbody>
				{#each perCity as row (row.city)}
					{@const baseCount = row.baseSites.length}
					{@const whatIfCount = row.whatIfSites.length}
					{@const delta = whatIfCount - baseCount}
					<tr class="border-b border-gray-100 hover:bg-gray-50">
						<td class="px-3 py-2 font-medium">{row.city}</td>
						<td class="px-3 py-2 text-right font-mono text-gray-500">{row.total}</td>
						<td class="px-3 py-2 text-right font-mono">{baseCount}</td>
						{#if totals.hasWhatIf}
							<td class="px-3 py-2 text-right font-mono">{whatIfCount}</td>
							<td
								class="px-3 py-2 text-right font-mono {delta === 0
									? 'text-gray-400'
									: delta > 0
									? 'text-emerald-700'
									: 'text-amber-700'}"
							>
								{delta > 0 ? '+' : ''}{delta}
							</td>
						{/if}
						<td class="px-3 py-2 text-xs">
							{#if !totals.hasWhatIf}
								<div class="flex flex-wrap gap-1">
									{#each row.baseSites as s (s.variable_symbol)}
										<span
											class="rounded border border-gray-300 bg-gray-50 px-1.5 py-0.5 text-[11px]"
											title={s.variable_symbol}
										>
											{siteName(s)}
										</span>
									{:else}
										<span class="text-gray-400">—</span>
									{/each}
								</div>
							{:else}
								<div class="flex flex-wrap gap-1">
									{#each row.whatIfSites as s (s.variable_symbol)}
										{@const isAdded = row.added.some((x) => x.variable_symbol === s.variable_symbol)}
										<span
											class="rounded border px-1.5 py-0.5 text-[11px] {isAdded
												? 'border-emerald-400 bg-emerald-50 text-emerald-800'
												: 'border-gray-300 bg-gray-50 text-gray-700'}"
											title={s.variable_symbol}
										>
											{isAdded ? '+ ' : ''}{siteName(s)}
										</span>
									{/each}
									{#each row.removed as s (s.variable_symbol)}
										<span
											class="rounded border border-amber-400 bg-amber-50 px-1.5 py-0.5 text-[11px] text-amber-800 line-through"
											title={s.variable_symbol}
										>
											− {siteName(s)}
										</span>
									{/each}
									{#if row.whatIfSites.length === 0 && row.removed.length === 0}
										<span class="text-gray-400">—</span>
									{/if}
								</div>
							{/if}
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
</div>
