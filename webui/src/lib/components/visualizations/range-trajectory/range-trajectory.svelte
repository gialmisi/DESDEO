<script lang="ts">
	/**
	 * Range trajectory chart
	 *
	 * @description
	 * Visualizes lower/upper bounds along navigation steps with optional markers for
	 * navigation points, reference points, and user bounds.
	 */
	import { onDestroy, onMount } from 'svelte';
	import * as d3 from 'd3';

	export let steps: number[] = [];
	export let lowerBounds: number[] = [];
	export let upperBounds: number[] = [];
	export let navigationPoints: number[] = [];
	export let referencePoints: Array<number | null> = [];
	export let boundPoints: Array<number | null> = [];
	export let label: string = '';
	export let color: string = '#4f8cff';
	export let height: number = 160;

	let width = 600;
	let svg: SVGSVGElement;
	let container: HTMLDivElement;
	let resizeObserver: ResizeObserver;

	function drawChart() {
		d3.select(svg).selectAll('*').remove();

		if (steps.length === 0 || lowerBounds.length === 0 || upperBounds.length === 0) {
			return;
		}

		const margin = { top: 16, right: 20, bottom: 30, left: 40 };
		const innerWidth = width - margin.left - margin.right;
		const innerHeight = height - margin.top - margin.bottom;

		const yMin = Math.min(...lowerBounds, ...upperBounds, ...navigationPoints);
		const yMax = Math.max(...lowerBounds, ...upperBounds, ...navigationPoints);

		const x = d3
			.scaleLinear()
			.domain([Math.min(...steps), Math.max(...steps)])
			.range([margin.left, margin.left + innerWidth]);

		const y = d3
			.scaleLinear()
			.domain([yMin, yMax])
			.nice()
			.range([margin.top + innerHeight, margin.top]);

		const area = d3
			.area<number>()
			.x((_, i) => x(steps[i]))
			.y0((_, i) => y(lowerBounds[i]))
			.y1((_, i) => y(upperBounds[i]));

		d3.select(svg)
			.append('path')
			.datum(steps)
			.attr('fill', color)
			.attr('opacity', 0.15)
			.attr('d', area);

		const lineLower = d3
			.line<number>()
			.x((_, i) => x(steps[i]))
			.y((_, i) => y(lowerBounds[i]));

		const lineUpper = d3
			.line<number>()
			.x((_, i) => x(steps[i]))
			.y((_, i) => y(upperBounds[i]));

		d3.select(svg)
			.append('path')
			.datum(steps)
			.attr('fill', 'none')
			.attr('stroke', color)
			.attr('stroke-width', 2)
			.attr('d', lineLower);

		d3.select(svg)
			.append('path')
			.datum(steps)
			.attr('fill', 'none')
			.attr('stroke', color)
			.attr('stroke-width', 2)
			.attr('stroke-dasharray', '6 4')
			.attr('d', lineUpper);

		if (navigationPoints.length === steps.length) {
			const navLine = d3
				.line<number>()
				.x((_, i) => x(steps[i]))
				.y((_, i) => y(navigationPoints[i]));

			d3.select(svg)
				.append('path')
				.datum(steps)
				.attr('fill', 'none')
				.attr('stroke', '#334155')
				.attr('stroke-width', 2)
				.attr('d', navLine);
		}

		const addMarkers = (values: Array<number | null>, markerColor: string, shape: 'circle' | 'square') => {
			values.forEach((value, i) => {
				if (value === null || Number.isNaN(value)) return;
				const cx = x(steps[i]);
				const cy = y(value);
				if (shape === 'circle') {
					d3.select(svg)
						.append('circle')
						.attr('cx', cx)
						.attr('cy', cy)
						.attr('r', 4)
						.attr('fill', markerColor);
				} else {
					d3.select(svg)
						.append('rect')
						.attr('x', cx - 4)
						.attr('y', cy - 4)
						.attr('width', 8)
						.attr('height', 8)
						.attr('fill', markerColor);
				}
			});
		};

		addMarkers(referencePoints, '#16a34a', 'circle');
		addMarkers(boundPoints, '#f97316', 'square');

		d3.select(svg)
			.append('g')
			.attr('transform', `translate(0,${margin.top + innerHeight})`)
			.call(d3.axisBottom(x).ticks(Math.min(6, steps.length)));

		d3.select(svg)
			.append('g')
			.attr('transform', `translate(${margin.left},0)`)
			.call(d3.axisLeft(y).ticks(4));

		d3.select(svg)
			.append('text')
			.attr('x', margin.left)
			.attr('y', 12)
			.attr('fill', '#111827')
			.attr('font-size', 12)
			.attr('font-weight', 600)
			.text(label);
	}

	onMount(() => {
		resizeObserver = new ResizeObserver((entries) => {
			for (const entry of entries) {
				width = entry.contentRect.width;
				drawChart();
			}
		});
		resizeObserver.observe(container);
		drawChart();
	});

	onDestroy(() => {
		resizeObserver.disconnect();
	});

	$: steps, lowerBounds, upperBounds, navigationPoints, referencePoints, boundPoints, label, height, drawChart();
</script>

<div bind:this={container} style="width: 100%; height: {height}px;">
	<svg bind:this={svg} style="width: 100%; height: 100%;" />
</div>
