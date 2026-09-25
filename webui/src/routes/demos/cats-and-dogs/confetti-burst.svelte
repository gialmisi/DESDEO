<script lang="ts">
	/**
	 * confetti-burst.svelte
	 *
	 * A one-shot burst of confetti, drawn on a canvas laid over the page. Used
	 * by the cats and dogs demo to make a small occasion of the moment the
	 * breed is revealed.
	 *
	 * The burst runs once when the component is mounted, so showing it inside
	 * an `{#if}` that becomes true on the reveal is all that is needed. The
	 * canvas ignores pointer events, so the buttons underneath stay usable.
	 *
	 * Nothing is drawn for a visitor who has asked for reduced motion.
	 */
	import { onMount } from 'svelte';
	import { COLOR_PALETTE } from '$lib/components/visualizations/utils/colors';

	let canvas: HTMLCanvasElement;

	const PARTICLE_COUNT = 400;
	const GRAVITY = 0.32;
	const DRAG = 0.99;
	const FADE_STARTS_AT = 0.65; // fraction of the run after which pieces fade
	const RUN_MS = 3000;

	type Piece = {
		x: number;
		y: number;
		vx: number;
		vy: number;
		width: number;
		height: number;
		angle: number;
		spin: number;
		color: string;
	};

	function makePieces(width: number, height: number): Piece[] {
		// The burst starts a little above the middle, roughly where the name of
		// the breed appears.
		const originX = width / 2;
		const originY = height * 0.38;

		return Array.from({ length: PARTICLE_COUNT }, () => {
			const angle = Math.random() * Math.PI * 2;
			// Spread the speeds so the burst has a dense core and plenty of
			// stragglers reaching the edges of the screen.
			const speed = 3 + Math.random() * 18;
			return {
				x: originX,
				y: originY,
				vx: Math.cos(angle) * speed,
				// Favour upward travel a little, so gravity has something to undo.
				vy: Math.sin(angle) * speed - 4,
				width: 6 + Math.random() * 6,
				height: 3 + Math.random() * 5,
				angle: Math.random() * Math.PI,
				spin: (Math.random() - 0.5) * 0.35,
				color: COLOR_PALETTE[Math.floor(Math.random() * COLOR_PALETTE.length)]
			};
		});
	}

	onMount(() => {
		const prefersReducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
		if (prefersReducedMotion) return;

		const context = canvas.getContext('2d');
		if (!context) return;

		let width = 0;
		let height = 0;

		function resize() {
			const ratio = window.devicePixelRatio || 1;
			width = window.innerWidth;
			height = window.innerHeight;
			canvas.width = width * ratio;
			canvas.height = height * ratio;
			context!.setTransform(ratio, 0, 0, ratio, 0, 0);
		}

		resize();
		window.addEventListener('resize', resize);

		const pieces = makePieces(width, height);
		const startedAt = performance.now();
		let frame = 0;

		function draw(now: number) {
			const elapsed = (now - startedAt) / RUN_MS;
			if (elapsed >= 1) {
				context!.clearRect(0, 0, width, height);
				return;
			}

			context!.clearRect(0, 0, width, height);
			context!.globalAlpha =
				elapsed < FADE_STARTS_AT ? 1 : 1 - (elapsed - FADE_STARTS_AT) / (1 - FADE_STARTS_AT);

			for (const piece of pieces) {
				piece.vx *= DRAG;
				piece.vy = piece.vy * DRAG + GRAVITY;
				piece.x += piece.vx;
				piece.y += piece.vy;
				piece.angle += piece.spin;

				context!.save();
				context!.translate(piece.x, piece.y);
				context!.rotate(piece.angle);
				context!.fillStyle = piece.color;
				// Squashing the height as the piece spins reads as a flake turning over.
				context!.fillRect(
					-piece.width / 2,
					-piece.height / 2,
					piece.width,
					piece.height * Math.abs(Math.cos(piece.angle))
				);
				context!.restore();
			}

			frame = requestAnimationFrame(draw);
		}

		frame = requestAnimationFrame(draw);

		return () => {
			cancelAnimationFrame(frame);
			window.removeEventListener('resize', resize);
		};
	});
</script>

<canvas bind:this={canvas} aria-hidden="true" class="pointer-events-none fixed inset-0 z-50"></canvas>
