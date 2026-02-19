/**
 * Utilities for the forest aerial map (Voronoi tessellation) visualization.
 * Each cluster is represented by an irregular polygon cell; additional
 * placeholder cells fill the remaining space like non-optimized forest stands.
 */

import { scaleLinear } from 'd3-scale';
import { interpolateRgb } from 'd3-interpolate';
import { Delaunay } from 'd3-delaunay';
import type { ClusterComparison } from '$lib/gen/models';

// ── Color scale (unchanged) ─────────────────────────────────────────

/**
 * Build a smooth 5-stop color scale: green -> lime -> yellow -> orange -> red.
 */
export function distanceColorScale(maxDistance: number) {
	const stops = [0, 0.25, 0.5, 0.75, 1].map((t) => t * maxDistance);
	const colors = ['#22c55e', '#84cc16', '#eab308', '#f97316', '#ef4444'];

	const scale = scaleLinear<string>()
		.domain(stops)
		.range(colors)
		.interpolate(interpolateRgb)
		.clamp(true);
	return scale;
}

/** Distinct stroke colors per cluster */
export const CLUSTER_BORDER_COLORS = ['#6366f1', '#8b5cf6', '#06b6d4', '#ec4899', '#f59e0b'];

/** CSS gradient string matching the 5-stop color scale */
export const DISTANCE_GRADIENT_CSS =
	'linear-gradient(to right, #22c55e, #84cc16, #eab308, #f97316, #ef4444)';

// ── Types ────────────────────────────────────────────────────────────

export interface VoronoiCell {
	path: string;
	centroid: { x: number; y: number };
	cluster: ClusterComparison | null;
	fillColor: string;
	strokeColor: string;
	label: string;
	interactive: boolean;
}

// ── Seeded PRNG ──────────────────────────────────────────────────────

function hashString(str: string): number {
	let h = 0;
	for (let i = 0; i < str.length; i++) {
		h = (Math.imul(31, h) + str.charCodeAt(i)) | 0;
	}
	return h >>> 0;
}

function seededRandom(seed: number): () => number {
	let s = seed | 0;
	return () => {
		s = (s + 0x6d2b79f5) | 0;
		let t = Math.imul(s ^ (s >>> 15), 1 | s);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

// ── Seed point generation ────────────────────────────────────────────

export interface SeedPoint {
	x: number;
	y: number;
	isCluster: boolean;
	clusterIndex: number; // -1 for placeholders
}

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

export function generateSeedPoints(
	clusterCount: number,
	clusterKeys: string[],
	width: number,
	height: number,
	placeholderCount: number = 10
): SeedPoint[] {
	const points: SeedPoint[] = [];
	const cx = width / 2;
	const cy = height / 2;
	const maxR = Math.min(width, height) * 0.3;

	// Cluster points: golden-angle spiral
	for (let i = 0; i < clusterCount; i++) {
		const r = maxR * Math.sqrt((i + 0.5) / clusterCount);
		const theta = i * GOLDEN_ANGLE;
		points.push({
			x: cx + r * Math.cos(theta),
			y: cy + r * Math.sin(theta),
			isCluster: true,
			clusterIndex: i
		});
	}

	// Deterministic seed from cluster keys
	const seedStr = clusterKeys.sort().join('|');
	const rng = seededRandom(hashString(seedStr));

	// Placeholder points: Poisson-disk-like rejection sampling
	const minDist = Math.min(width, height) * 0.08;
	const margin = 20;
	let attempts = 0;
	const maxAttempts = placeholderCount * 50;

	while (points.filter((p) => !p.isCluster).length < placeholderCount && attempts < maxAttempts) {
		attempts++;
		const px = margin + rng() * (width - 2 * margin);
		const py = margin + rng() * (height - 2 * margin);

		// Check minimum distance from all existing points
		let tooClose = false;
		for (const p of points) {
			const dx = p.x - px;
			const dy = p.y - py;
			if (dx * dx + dy * dy < minDist * minDist) {
				tooClose = true;
				break;
			}
		}
		if (!tooClose) {
			points.push({ x: px, y: py, isCluster: false, clusterIndex: -1 });
		}
	}

	return points;
}

// ── Geometry helpers ─────────────────────────────────────────────────

export function polygonToSvgPath(polygon: ArrayLike<[number, number]>): string {
	if (!polygon || polygon.length === 0) return '';
	const parts: string[] = [`M${polygon[0][0]},${polygon[0][1]}`];
	for (let i = 1; i < polygon.length; i++) {
		parts.push(`L${polygon[i][0]},${polygon[i][1]}`);
	}
	parts.push('Z');
	return parts.join('');
}

export function polygonCentroid(polygon: ArrayLike<[number, number]>): { x: number; y: number } {
	let sx = 0;
	let sy = 0;
	const n = polygon.length;
	for (let i = 0; i < n; i++) {
		sx += polygon[i][0];
		sy += polygon[i][1];
	}
	return { x: sx / n, y: sy / n };
}

// ── Cell construction ────────────────────────────────────────────────

const PLACEHOLDER_FILLS = ['#d4d9d0', '#c8cec4'];

export function buildVoronoiCells(
	seedPoints: SeedPoint[],
	clusters: ClusterComparison[],
	colorScale: (d: number) => string,
	width: number,
	height: number
): VoronoiCell[] {
	if (seedPoints.length === 0) return [];

	const coords: [number, number][] = seedPoints.map((p) => [p.x, p.y]);
	const delaunay = Delaunay.from(coords);
	const voronoi = delaunay.voronoi([0, 0, width, height]);

	const cells: VoronoiCell[] = [];

	for (let i = 0; i < seedPoints.length; i++) {
		const polygon = voronoi.cellPolygon(i);
		if (!polygon) continue;

		const sp = seedPoints[i];
		const path = polygonToSvgPath(polygon);
		const centroid = polygonCentroid(polygon);

		if (sp.isCluster && sp.clusterIndex >= 0 && sp.clusterIndex < clusters.length) {
			const cluster = clusters[sp.clusterIndex];
			const ci = sp.clusterIndex;
			cells.push({
				path,
				centroid,
				cluster,
				fillColor: colorScale(cluster.distance),
				strokeColor: CLUSTER_BORDER_COLORS[ci % CLUSTER_BORDER_COLORS.length],
				label: cluster.cluster_key
					.replace('_', ' ')
					.replace(/\b\w/g, (c) => c.toUpperCase()),
				interactive: true
			});
		} else {
			cells.push({
				path,
				centroid,
				cluster: null,
				fillColor: PLACEHOLDER_FILLS[i % 2],
				strokeColor: '#b8bfb4',
				label: '',
				interactive: false
			});
		}
	}

	return cells;
}
