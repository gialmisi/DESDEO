/**
 * Utilities for the honeycomb (hex grid) visualization.
 * Each cluster is represented by a single large hexagon.
 */

import { scaleLinear } from 'd3-scale';
import { interpolateRgb } from 'd3-interpolate';

/** Hex center coordinates */
export interface HexPosition {
	x: number;
	y: number;
}

/**
 * Generate SVG path data for a flat-top regular hexagon centered at (cx, cy).
 */
export function hexPath(cx: number, cy: number, radius: number): string {
	const points: string[] = [];
	for (let i = 0; i < 6; i++) {
		const angle = (Math.PI / 3) * i;
		const px = cx + radius * Math.cos(angle);
		const py = cy + radius * Math.sin(angle);
		points.push(`${px},${py}`);
	}
	return `M${points.join('L')}Z`;
}

/**
 * Compute positions for N cluster hexagons arranged horizontally with gaps.
 */
export function computeClusterPositions(
	count: number,
	hexRadius: number,
	containerWidth: number,
	containerHeight: number,
	gap: number = 30
): { positions: HexPosition[]; labelPositions: HexPosition[] } {
	if (count === 0) return { positions: [], labelPositions: [] };

	const hexW = hexRadius * Math.sqrt(3);
	const totalWidth = count * hexW + (count - 1) * gap;
	const startX = (containerWidth - totalWidth) / 2 + hexW / 2;
	const centerY = containerHeight / 2;

	const positions: HexPosition[] = [];
	const labelPositions: HexPosition[] = [];

	for (let i = 0; i < count; i++) {
		const x = startX + i * (hexW + gap);
		positions.push({ x, y: centerY });
		labelPositions.push({ x, y: centerY - hexRadius - 12 });
	}

	return { positions, labelPositions };
}

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
