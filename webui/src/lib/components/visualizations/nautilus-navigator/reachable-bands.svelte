<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	export let objectiveSymbols: string[] = [];
	export let objectiveLabels: Record<string, string> = {};
	export let lowerBounds: Record<string, number[]> = {};
	export let upperBounds: Record<string, number[]> = {};
	export let navigationPoints: Record<string, number[]> = {};
	export let aspirationLevels: Record<string, number> = {};
	export let stepCursor = 0;

	const dispatch = createEventDispatcher<{ stepSelect: number }>();

	const chartWidth = 320;
	const chartHeight = 140;

	const getSeriesLength = (symbol: string) => lowerBounds[symbol]?.length ?? 0;

	const getRange = (symbol: string) => {
		const lower = lowerBounds[symbol] ?? [];
		const upper = upperBounds[symbol] ?? [];
		const combined = [...lower, ...upper, aspirationLevels[symbol], ...(navigationPoints[symbol] ?? [])].filter(
			(v) => typeof v === 'number' && !Number.isNaN(v)
		) as number[];
		const min = combined.length ? Math.min(...combined) : 0;
		const max = combined.length ? Math.max(...combined) : 1;
		return min === max ? [min - 1, max + 1] : [min, max];
	};

	const toX = (index: number, length: number) =>
		length <= 1 ? 0 : (index / (length - 1)) * chartWidth;

	const toY = (value: number, min: number, max: number) =>
		chartHeight - ((value - min) / (max - min)) * chartHeight;

	const seriesToPath = (series: number[], min: number, max: number) => {
		if (!series.length) return '';
		return series
			.map((value, index) => `${index === 0 ? 'M' : 'L'} ${toX(index, series.length)} ${toY(value, min, max)}`)
			.join(' ');
	};

	const bandToPath = (lower: number[], upper: number[], min: number, max: number) => {
		if (!lower.length || !upper.length) return '';
		const upperPath = upper
			.map((value, index) => `${index === 0 ? 'M' : 'L'} ${toX(index, upper.length)} ${toY(value, min, max)}`)
			.join(' ');
		const lowerPath = lower
			.slice()
			.reverse()
			.map((value, index) => `L ${toX(lower.length - 1 - index, lower.length)} ${toY(value, min, max)}`)
			.join(' ');
		return `${upperPath} ${lowerPath} Z`;
	};

	const handleClick = (event: MouseEvent, symbol: string) => {
		const length = getSeriesLength(symbol);
		if (length <= 1) return;
		const rect = (event.currentTarget as SVGSVGElement).getBoundingClientRect();
		const x = event.clientX - rect.left;
		const index = Math.round((x / rect.width) * (length - 1));
		dispatch('stepSelect', Math.max(0, Math.min(length - 1, index)));
	};

	const handleKeydown = (event: KeyboardEvent, symbol: string) => {
		if (event.key !== 'Enter' && event.key !== ' ') return;
		event.preventDefault();
		dispatch('stepSelect', Math.min(stepCursor, Math.max(0, getSeriesLength(symbol) - 1)));
	};
</script>

<div class="grid gap-4">
	{#if objectiveSymbols.length === 0}
		<p class="text-muted-foreground">No navigation data yet.</p>
	{:else}
		{#each objectiveSymbols as symbol}
			{#key symbol}
				<div class="rounded-lg border bg-white p-3 shadow-sm">
					<div class="mb-2 flex items-center justify-between text-sm font-semibold">
						<span>{objectiveLabels[symbol] ?? symbol}</span>
						<span class="text-xs text-muted-foreground">Step {stepCursor + 1}</span>
					</div>
					{#if getSeriesLength(symbol) > 0}
						{#key getSeriesLength(symbol)}
							{@const [minValue, maxValue] = getRange(symbol)}
							<svg
								class="h-36 w-full cursor-pointer"
								viewBox={`0 0 ${chartWidth} ${chartHeight}`}
								on:click={(event) => handleClick(event, symbol)}
								on:keydown={(event) => handleKeydown(event, symbol)}
								role="button"
								tabindex="0"
								aria-label={`Reachable bounds for ${objectiveLabels[symbol] ?? symbol}`}
							>
								<rect width={chartWidth} height={chartHeight} fill="#f8fafc" rx="8" />
								<path
									d={bandToPath(lowerBounds[symbol] ?? [], upperBounds[symbol] ?? [], minValue, maxValue)}
									fill="#bfdbfe"
									opacity="0.7"
								/>
								<path
									d={seriesToPath(lowerBounds[symbol] ?? [], minValue, maxValue)}
									fill="none"
									stroke="#2563eb"
									stroke-width="2"
								/>
								<path
									d={seriesToPath(upperBounds[symbol] ?? [], minValue, maxValue)}
									fill="none"
									stroke="#1d4ed8"
									stroke-width="2"
								/>
								{#if aspirationLevels[symbol] !== undefined}
									<line
										x1="0"
										x2={chartWidth}
										y1={toY(aspirationLevels[symbol], minValue, maxValue)}
										y2={toY(aspirationLevels[symbol], minValue, maxValue)}
										stroke="#f59e0b"
										stroke-width="2"
										stroke-dasharray="4 4"
									/>
								{/if}
								{#if navigationPoints[symbol]?.[stepCursor] !== undefined}
									<circle
										cx={toX(stepCursor, getSeriesLength(symbol))}
										cy={toY(navigationPoints[symbol][stepCursor], minValue, maxValue)}
										r="4"
										fill="#dc2626"
									/>
								{/if}
							</svg>
							<div class="mt-2 flex items-center justify-between text-xs text-muted-foreground">
								<span>{minValue.toFixed(2)}</span>
								<span>{maxValue.toFixed(2)}</span>
							</div>
						{/key}
					{:else}
						<p class="text-xs text-muted-foreground">No reachable bounds calculated yet.</p>
					{/if}
				</div>
			{/key}
		{/each}
	{/if}
</div>
