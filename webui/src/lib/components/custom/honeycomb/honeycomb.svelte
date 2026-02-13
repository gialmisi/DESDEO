<script lang="ts">
	import { onMount } from 'svelte';
	import type { ScenarioComparisonResponse, ClusterComparison } from '$lib/gen/models';
	import {
		computeClusterPositions,
		hexPath,
		distanceColorScale,
		CLUSTER_BORDER_COLORS
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

	let hexRadius = $derived(Math.min(width / 8, height / 3, 80));
	// Fixed scale so that each hex's color depends only on its own absolute
	// distance, not on other clusters.  The backend distance is a normalized
	// Euclidean metric (each objective diff / ideal-nadir range), so 1.0 is
	// already a large disagreement.  Clamping keeps the scale stable when an
	// individual owner re-solves.
	let colorScale = $derived(distanceColorScale(1));

	let clusterHexes = $derived.by(() => {
		if (!data || !data.clusters.length || hexRadius <= 0 || width <= 0) return [];

		const { positions, labelPositions } = computeClusterPositions(
			data.clusters.length,
			hexRadius,
			width,
			height,
			hexRadius * 0.8
		);

		return data.clusters.map((cluster, i) => ({
			cluster,
			pos: positions[i],
			labelPos: labelPositions[i],
			fillColor: colorScale(cluster.distance),
			strokeColor: CLUSTER_BORDER_COLORS[i % CLUSTER_BORDER_COLORS.length],
			path: hexPath(positions[i].x, positions[i].y, hexRadius * 0.92),
			label: cluster.cluster_key
				.replace('_', ' ')
				.replace(/\b\w/g, (c) => c.toUpperCase())
		}));
	});

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
	{#if data && clusterHexes.length > 0}
		<svg {width} {height}>
			{#each clusterHexes as hex}
				<!-- Cluster label above hex -->
				<text
					x={hex.labelPos.x}
					y={hex.labelPos.y}
					text-anchor="middle"
					class="select-none fill-gray-600 text-sm font-semibold"
					font-size={hexRadius * 0.22}
				>
					{hex.label}
				</text>

				<!-- Cluster hex -->
				<path
					d={hex.path}
					fill={hex.fillColor}
					stroke={hex.strokeColor}
					stroke-width={selectedClusterKeys.includes(hex.cluster.cluster_key) ? 5 : 3}
					class="cursor-pointer transition-opacity hover:opacity-80"
					role="img"
					aria-label="{hex.label} cluster"
					onmouseenter={(e) => showTooltip(e, hex.cluster)}
					onmousemove={moveTooltip}
					onmouseleave={hideTooltip}
					onclick={() => onclusterclick?.(hex.cluster)}
				/>

				<!-- Distance value inside hex -->
				<text
					x={hex.pos.x}
					y={hex.pos.y}
					text-anchor="middle"
					dominant-baseline="central"
					class="pointer-events-none select-none fill-white font-bold"
					font-size={hexRadius * 0.28}
				>
					{formatNumber(hex.cluster.distance, 3)}
				</text>
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
