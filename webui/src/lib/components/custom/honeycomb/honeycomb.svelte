<script lang="ts">
	import { onMount } from 'svelte';
	import type { ScenarioComparisonResponse, ClusterComparison } from '$lib/gen/models';
	import {
		generateSeedPoints,
		buildVoronoiCells,
		distanceColorScale,
		CLUSTER_BORDER_COLORS,
		type VoronoiCell
	} from './honeycomb-utils';
	import { formatNumber } from '$lib/helpers';

	interface Props {
		data: ScenarioComparisonResponse | null;
		onclusterclick?: (cluster: ClusterComparison) => void;
		selectedClusterKeys?: string[];
	}

	let { data, onclusterclick, selectedClusterKeys = [] }: Props = $props();

	let container: HTMLDivElement | undefined = $state();
	let width = $state(0);
	let height = $state(0);

	let tooltip = $state<{
		visible: boolean;
		x: number;
		y: number;
		cluster: ClusterComparison | null;
	}>({ visible: false, x: 0, y: 0, cluster: null });

	onMount(() => {
		if (!container) return;
		const observer = new ResizeObserver(() => {
			if (container) {
				width = container.clientWidth;
				height = container.clientHeight;
			}
		});
		observer.observe(container);
		return () => observer.disconnect();
	});

	// Fixed scale: 1.0 is already a large disagreement
	let colorScale = $derived(distanceColorScale(1));

	let voronoiCells = $derived.by(() => {
		if (!data || !data.clusters.length || width <= 0 || height <= 0) return [];

		const clusterKeys = data.clusters.map((c) => c.cluster_key);
		const seedPoints = generateSeedPoints(data.clusters.length, clusterKeys, width, height);
		return buildVoronoiCells(seedPoints, data.clusters, colorScale, width, height);
	});

	let placeholderCells = $derived(voronoiCells.filter((c) => !c.interactive));
	let clusterCells = $derived(voronoiCells.filter((c) => c.interactive));

	function showTooltip(e: MouseEvent, cluster: ClusterComparison) {
		tooltip = { visible: true, x: e.clientX, y: e.clientY, cluster };
	}

	function moveTooltip(e: MouseEvent) {
		if (tooltip.visible) {
			tooltip = { ...tooltip, x: e.clientX, y: e.clientY };
		}
	}

	function hideTooltip() {
		tooltip = { ...tooltip, visible: false, cluster: null };
	}

	// Smart positioning: flip to left side if too close to right edge
	let tooltipLeft = $derived.by(() => {
		const tooltipW = 300;
		if (tooltip.x + tooltipW + 20 > window.innerWidth) {
			return tooltip.x - tooltipW - 12;
		}
		return tooltip.x + 12;
	});
</script>

<div class="absolute inset-0" bind:this={container}>
	{#if data && voronoiCells.length > 0}
		<svg {width} {height}>
			<!-- Forest texture pattern -->
			<defs>
				<pattern id="forest-texture" width="20" height="20" patternUnits="userSpaceOnUse">
					<circle cx="5" cy="5" r="1.5" fill="#a0a89c" opacity="0.3" />
					<circle cx="15" cy="12" r="1" fill="#939b8f" opacity="0.25" />
					<circle cx="10" cy="18" r="1.2" fill="#a8b0a4" opacity="0.2" />
				</pattern>
			</defs>

			<!-- Layer 1: Placeholder cells (background) -->
			{#each placeholderCells as cell}
				<path
					d={cell.path}
					fill={cell.fillColor}
					stroke={cell.strokeColor}
					stroke-width="1"
				/>
				<path d={cell.path} fill="url(#forest-texture)" />
			{/each}

			<!-- Layer 2: Cluster cells (foreground) -->
			{#each clusterCells as cell}
				{@const isSelected = cell.cluster && selectedClusterKeys.includes(cell.cluster.cluster_key)}
				<path
					d={cell.path}
					fill={cell.fillColor}
					stroke={cell.strokeColor}
					stroke-width="2"
					class="cursor-pointer transition-opacity hover:opacity-80"
					role="img"
					aria-label="{cell.label} cluster"
					onmouseenter={(e) => cell.cluster && showTooltip(e, cell.cluster)}
					onmousemove={moveTooltip}
					onmouseleave={hideTooltip}
					onclick={() => cell.cluster && onclusterclick?.(cell.cluster)}
				/>

				<!-- Cluster label -->
				<text
					x={cell.centroid.x}
					y={cell.centroid.y - 8}
					text-anchor="middle"
					dominant-baseline="central"
					class="pointer-events-none select-none font-semibold"
					font-size="12"
					fill="white"
					style="text-shadow: 0 1px 3px rgba(0,0,0,0.5);"
				>
					{cell.label}
				</text>

				<!-- Distance value -->
				<text
					x={cell.centroid.x}
					y={cell.centroid.y + 10}
					text-anchor="middle"
					dominant-baseline="central"
					class="pointer-events-none select-none font-bold"
					font-size="14"
					fill="white"
					style="text-shadow: 0 1px 3px rgba(0,0,0,0.5);"
				>
					{cell.cluster ? formatNumber(cell.cluster.distance, 3) : ''}
				</text>
			{/each}

			<!-- Layer 3: Selection highlights (top) -->
			{#each clusterCells as cell, i}
				{#if cell.cluster && selectedClusterKeys.includes(cell.cluster.cluster_key)}
					<path
						d={cell.path}
						fill="none"
						stroke={CLUSTER_BORDER_COLORS[i % CLUSTER_BORDER_COLORS.length]}
						stroke-width="4"
						stroke-dasharray="8 4"
						class="pointer-events-none"
					/>
				{/if}
			{/each}
		</svg>
	{:else}
		<div class="flex h-full items-center justify-center text-sm text-gray-400">
			No comparison data available.
		</div>
	{/if}

	{#if tooltip.visible && tooltip.cluster}
		{@const c = tooltip.cluster}
		<div
			class="pointer-events-none fixed z-[9999] rounded-md border border-gray-200 bg-white px-3 py-2 text-xs text-gray-800 shadow-lg"
			style="left: {tooltipLeft}px; top: {tooltip.y - 10}px; max-width: 300px;"
		>
			<div class="mb-1 font-bold text-gray-900">
				{c.cluster_key.replace('_', ' ').replace(/\b\w/g, (ch) => ch.toUpperCase())}
			</div>
			<table class="w-full">
				<thead>
					<tr class="border-b border-gray-200">
						<th class="pr-2 text-left font-medium text-gray-500">Objective</th>
						<th class="pr-2 text-right font-medium text-gray-500">Supervisor</th>
						<th class="pr-2 text-right font-medium text-gray-500">Owner</th>
						<th class="text-right font-medium text-gray-500">Delta</th>
					</tr>
				</thead>
				<tbody>
					{#each Object.keys(c.supervisor_objectives) as key}
						{@const sv = c.supervisor_objectives[key]}
						{@const ov = c.owner_objectives[key]}
						{@const delta = sv - ov}
						<tr>
							<td class="pr-2 text-gray-700">{key}</td>
							<td class="pr-2 text-right">{formatNumber(sv, 1)}</td>
							<td class="pr-2 text-right">{formatNumber(ov, 1)}</td>
							<td
								class="text-right"
								class:text-red-600={delta < 0}
								class:text-green-600={delta > 0}
							>
								{delta >= 0 ? '+' : ''}{formatNumber(delta, 1)}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
			<div class="mt-1 text-gray-400">Distance: {formatNumber(c.distance, 3)}</div>
		</div>
	{/if}
</div>
