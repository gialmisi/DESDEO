<script lang="ts">
	import { onMount } from 'svelte';
	import type { SiteInfo, SolverResults, VariableFixing } from '$lib/gen/models';
	import { buildMapSiteSelectionMapPost } from '$lib/gen/endpoints/DESDEOFastAPI';

	interface SiteSelectionMapNode {
		name: string;
		lat: number;
		lon: number;
		size: number;
		color: string;
		tooltip: string;
	}

	interface SiteSelectionMapEdge {
		from_lat: number;
		from_lon: number;
		to_lat: number;
		to_lon: number;
	}

	interface SiteSelectionMapResponse {
		nodes: SiteSelectionMapNode[];
		edges: SiteSelectionMapEdge[];
		center: [number, number];
		site_variable_symbols?: string[];
		site_node_names?: string[];
	}

	type ConstraintState = 'free' | 'restricted' | 'forced';

	interface Props {
		problem_id: number;
		solution: SolverResults;
		sites?: SiteInfo[];
		siteStates?: Map<string, ConstraintState>;
		on_constraints_changed?: (fixings: VariableFixing[]) => void;
	}

	let {
		problem_id,
		solution,
		sites = [],
		siteStates = $bindable(new Map()),
		on_constraints_changed,
	}: Props = $props();

	let mapContainer = $state<HTMLDivElement>();
	let map: L.Map | null = null;
	let loading = $state(true);
	let error = $state<string | null>(null);
	let mapData = $state<SiteSelectionMapResponse | null>(null);
	// City whose per-site picker panel is open; null when closed.
	let openCity = $state<string | null>(null);
	let markerLayer: L.LayerGroup | null = null;

	const COLOR_RESTRICTED = '#EF4444';
	const COLOR_FORCED = '#22C55E';
	const COLOR_MIXED = '#A855F7';

	let fixings = $derived.by(() => {
		const result: VariableFixing[] = [];
		for (const [symbol, state] of siteStates) {
			if (state === 'restricted') {
				result.push({ variable_symbol: symbol, fixed_value: 0 });
			} else if (state === 'forced') {
				result.push({ variable_symbol: symbol, fixed_value: 1 });
			}
		}
		return result;
	});

	$effect(() => {
		if (on_constraints_changed) {
			on_constraints_changed(fixings);
		}
	});

	// Group provided sites by their host city.
	let sitesByCity = $derived.by(() => {
		const grouped = new Map<string, SiteInfo[]>();
		for (const s of sites) {
			const list = grouped.get(s.node);
			if (list) list.push(s);
			else grouped.set(s.node, [s]);
		}
		return grouped;
	});

	// Fall back to map-derived symbols for cities without explicit site metadata.
	let symbolsByCityFallback = $derived.by(() => {
		const grouped = new Map<string, string[]>();
		const data = mapData;
		if (data?.site_variable_symbols && data?.site_node_names) {
			for (let i = 0; i < data.site_variable_symbols.length; i++) {
				const node = data.site_node_names[i];
				const list = grouped.get(node);
				if (list) list.push(data.site_variable_symbols[i]);
				else grouped.set(node, [data.site_variable_symbols[i]]);
			}
		}
		return grouped;
	});

	function symbolsForCity(cityName: string): string[] {
		const fromSites = sitesByCity.get(cityName);
		if (fromSites && fromSites.length > 0) return fromSites.map((s) => s.variable_symbol);
		return symbolsByCityFallback.get(cityName) ?? [];
	}

	function aggregateState(symbols: string[]): {
		state: ConstraintState | 'mixed' | 'partial-restricted' | 'partial-forced';
		forced: number;
		restricted: number;
	} {
		let forced = 0;
		let restricted = 0;
		for (const sym of symbols) {
			const s = siteStates.get(sym);
			if (s === 'forced') forced++;
			else if (s === 'restricted') restricted++;
		}
		const total = symbols.length;
		if (forced === 0 && restricted === 0) return { state: 'free', forced, restricted };
		if (forced > 0 && restricted > 0) return { state: 'mixed', forced, restricted };
		if (forced === total) return { state: 'forced', forced, restricted };
		if (restricted === total) return { state: 'restricted', forced, restricted };
		if (forced > 0) return { state: 'partial-forced', forced, restricted };
		return { state: 'partial-restricted', forced, restricted };
	}

	function aggregateColor(node: SiteSelectionMapNode, symbols: string[]): string {
		const agg = aggregateState(symbols);
		if (agg.state === 'forced' || agg.state === 'partial-forced') return COLOR_FORCED;
		if (agg.state === 'restricted' || agg.state === 'partial-restricted') return COLOR_RESTRICTED;
		if (agg.state === 'mixed') return COLOR_MIXED;
		return node.color;
	}

	function cycleSite(symbol: string) {
		const current = siteStates.get(symbol) ?? 'free';
		const next: ConstraintState = current === 'free' ? 'restricted' : current === 'restricted' ? 'forced' : 'free';
		const nextMap = new Map(siteStates);
		if (next === 'free') nextMap.delete(symbol);
		else nextMap.set(symbol, next);
		siteStates = nextMap;
	}

	function setCitySites(cityName: string, target: ConstraintState | 'free') {
		const symbols = symbolsForCity(cityName);
		const nextMap = new Map(siteStates);
		for (const sym of symbols) {
			if (target === 'free') nextMap.delete(sym);
			else nextMap.set(sym, target);
		}
		siteStates = nextMap;
	}

	function siteIsVisited(symbol: string): boolean {
		const vars = solution?.optimal_variables as Record<string, unknown> | undefined;
		if (!vars) return false;
		const v = vars[symbol];
		if (Array.isArray(v)) return Math.round(Number(v[0])) === 1;
		if (v != null) return Math.round(Number(v)) === 1;
		return false;
	}

	async function fetchMapData(): Promise<SiteSelectionMapResponse> {
		const res = await buildMapSiteSelectionMapPost({
			problem_id,
			optimal_variables: solution.optimal_variables as Record<string, unknown>
		});
		if (res.status === 404) {
			throw new Error('No site selection map metadata configured for this problem.');
		}
		if (res.status !== 200) {
			throw new Error(`Map API error: ${res.status}`);
		}
		return res.data as SiteSelectionMapResponse;
	}

	async function renderMap(data: SiteSelectionMapResponse) {
		const L = await import('leaflet');

		if (map) {
			map.remove();
			map = null;
		}

		map = L.map(mapContainer!, {
			center: data.center as [number, number],
			zoom: 9
		});

		L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
			attribution:
				'&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
			subdomains: 'abcd',
			maxZoom: 19
		}).addTo(map);

		for (const edge of data.edges) {
			L.polyline(
				[
					[edge.from_lat, edge.from_lon],
					[edge.to_lat, edge.to_lon]
				],
				{ color: 'black', weight: 1.5, opacity: 0.5 }
			).addTo(map);
		}

		markerLayer = L.layerGroup().addTo(map);
		updateMarkers(data, L);

		const bounds: L.LatLngExpression[] = data.nodes.map((n) => [n.lat, n.lon]);
		if (bounds.length > 0) {
			map.fitBounds(L.latLngBounds(bounds), { padding: [20, 20] });
		}

		const legend = new L.Control({ position: 'bottomright' });
		legend.onAdd = () => {
			const div = L.DomUtil.create('div', 'leaflet-legend');
			div.innerHTML = `
				<div style="background:white; padding:8px 12px; border-radius:6px; box-shadow:0 1px 4px rgba(0,0,0,.3); font-size:12px; line-height:20px;">
					<div style="font-weight:600; margin-bottom:4px;">Legend</div>
					<div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#FFA500;border:1px solid black;vertical-align:middle;margin-right:6px;"></span>Has active sites</div>
					<div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#FFFF00;border:1px solid black;vertical-align:middle;margin-right:6px;"></span>Covered by nearby sites</div>
					<div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#808080;border:1px solid black;vertical-align:middle;margin-right:6px;"></span>Not covered</div>
					<div style="margin-top:4px;"><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:${COLOR_RESTRICTED};border:1px solid black;vertical-align:middle;margin-right:6px;"></span>Restricted sites</div>
					<div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:${COLOR_FORCED};border:1px solid black;vertical-align:middle;margin-right:6px;"></span>Forced sites</div>
					<div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:${COLOR_MIXED};border:1px solid black;vertical-align:middle;margin-right:6px;"></span>Mixed (some forced + some restricted)</div>
					<div style="margin-top:4px;"><span style="display:inline-block;width:20px;height:2px;background:black;vertical-align:middle;margin-right:6px;"></span>Coverage link</div>
					<div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:rgba(128,128,128,0.25);border:1px dashed #888;vertical-align:middle;margin-right:6px;"></span>No candidate sites</div>
				</div>`;
			return div;
		};
		legend.addTo(map);
	}

	function updateMarkers(data: SiteSelectionMapResponse, L: typeof import('leaflet')) {
		if (!markerLayer) return;
		markerLayer.clearLayers();

		for (const node of data.nodes) {
			const symbols = symbolsForCity(node.name);
			const hasSites = symbols.length > 0;
			const fillColor = aggregateColor(node, symbols);
			const agg = aggregateState(symbols);

			const marker = L.circleMarker([node.lat, node.lon], {
				radius: node.size,
				color: hasSites ? 'black' : '#888',
				weight: 1,
				dashArray: hasSites ? undefined : '3 3',
				fillColor,
				fillOpacity: hasSites ? 0.7 : 0.25
			});

			if (agg.forced + agg.restricted > 0) {
				const badge = L.divIcon({
					className: 'site-constraint-badge',
					html: `<span style="font-size:10px;font-weight:700;color:white;text-shadow:0 0 3px black;background:rgba(0,0,0,0.55);border-radius:8px;padding:1px 5px;">${agg.forced + agg.restricted}</span>`,
					iconSize: [22, 14],
					iconAnchor: [11, 7]
				});
				L.marker([node.lat, node.lon], { icon: badge, interactive: false }).addTo(markerLayer);
			}

			const tip = node.tooltip + (hasSites
				? `<br><i>Click to manage ${symbols.length} site(s)</i>`
				: `<br><i>No candidate sites in this city</i>`);
			marker.bindTooltip(tip, { direction: 'top', offset: [0, -5] });

			if (hasSites) {
				marker.on('click', () => {
					openCity = node.name;
				});
				marker.getElement?.()?.style.setProperty('cursor', 'pointer');
			}

			marker.addTo(markerLayer);
		}
	}

	$effect(() => {
		// Re-read state to register reactivity.
		void siteStates;
		if (mapData && markerLayer) {
			import('leaflet').then((L) => updateMarkers(mapData!, L));
		}
	});

	async function loadMap() {
		loading = true;
		error = null;
		try {
			const data = await fetchMapData();
			mapData = data;
			await renderMap(data);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			console.error('Site selection map error:', e);
		} finally {
			loading = false;
		}
	}

	onMount(() => {
		loadMap();
		return () => {
			if (map) {
				map.remove();
				map = null;
			}
		};
	});

	$effect(() => {
		if (solution && mapContainer) {
			loadMap();
		}
	});

	export function clearConstraints() {
		siteStates = new Map();
		openCity = null;
	}

	let openCitySites = $derived.by(() => {
		if (!openCity) return [] as SiteInfo[];
		const fromSites = sitesByCity.get(openCity);
		if (fromSites && fromSites.length > 0) return fromSites;
		// Fallback: synthesize SiteInfo entries from map data symbols.
		const fallbackSyms = symbolsByCityFallback.get(openCity) ?? [];
		return fallbackSyms.map((sym, i) => ({
			index: i,
			name: sym,
			node: openCity!,
			lat: 0,
			lon: 0,
			variable_symbol: sym
		}));
	});

	function stateLabel(s: ConstraintState): string {
		if (s === 'forced') return 'Forced in';
		if (s === 'restricted') return 'Excluded';
		return 'Free';
	}

	function stateBadgeClass(s: ConstraintState): string {
		if (s === 'forced') return 'bg-green-100 text-green-800 border-green-300';
		if (s === 'restricted') return 'bg-red-100 text-red-800 border-red-300';
		return 'bg-gray-100 text-gray-700 border-gray-300';
	}
</script>

<svelte:head>
	<link
		rel="stylesheet"
		href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
		integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
		crossorigin=""
	/>
</svelte:head>

<div class="relative h-full w-full">
	{#if loading}
		<div class="flex h-full items-center justify-center text-sm text-gray-400">
			Loading map...
		</div>
	{/if}
	{#if error}
		<div class="flex h-full items-center justify-center text-sm text-red-400">
			{error}
		</div>
	{/if}
	<div bind:this={mapContainer} class="h-full w-full" class:invisible={loading || !!error}></div>

	{#if openCity}
		{@const citySites = openCitySites}
		<div
			class="absolute top-3 right-3 z-[1000] w-80 max-h-[85%] overflow-y-auto rounded-md border border-gray-200 bg-white shadow-lg"
		>
			<div class="flex items-center justify-between border-b border-gray-200 bg-gray-50 px-3 py-2">
				<div>
					<div class="text-sm font-semibold">{openCity}</div>
					<div class="text-xs text-gray-500">
						{citySites.length} site{citySites.length === 1 ? '' : 's'}
					</div>
				</div>
				<button
					class="text-gray-500 hover:text-gray-800"
					aria-label="Close site picker"
					onclick={() => (openCity = null)}
				>
					&times;
				</button>
			</div>
			<div class="flex gap-2 border-b border-gray-200 px-3 py-2 text-xs">
				<button
					class="rounded border border-green-300 bg-green-50 px-2 py-1 text-green-700 hover:bg-green-100"
					onclick={() => openCity && setCitySites(openCity, 'forced')}
				>Force all</button>
				<button
					class="rounded border border-red-300 bg-red-50 px-2 py-1 text-red-700 hover:bg-red-100"
					onclick={() => openCity && setCitySites(openCity, 'restricted')}
				>Exclude all</button>
				<button
					class="rounded border border-gray-300 bg-gray-50 px-2 py-1 text-gray-700 hover:bg-gray-100"
					onclick={() => openCity && setCitySites(openCity, 'free')}
				>Clear</button>
			</div>
			<ul class="divide-y divide-gray-100">
				{#each citySites as site (site.variable_symbol)}
					{@const s = siteStates.get(site.variable_symbol) ?? 'free'}
					{@const visited = siteIsVisited(site.variable_symbol)}
					<li class="flex items-center justify-between gap-2 px-3 py-2">
						<div class="min-w-0 flex-1">
							<div class="truncate text-sm font-medium" title={site.name}>{site.name}</div>
							<div class="text-[11px] text-gray-500">
								{visited ? 'Currently visited' : 'Not visited'} &middot; {site.variable_symbol}
							</div>
						</div>
						<button
							class={`rounded border px-2 py-1 text-[11px] ${stateBadgeClass(s)}`}
							onclick={() => cycleSite(site.variable_symbol)}
							title="Click to cycle: free → excluded → forced → free"
						>
							{stateLabel(s)}
						</button>
					</li>
				{/each}
			</ul>
		</div>
	{/if}
</div>
