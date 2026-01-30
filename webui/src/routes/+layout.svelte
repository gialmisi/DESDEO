<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import { appContext } from '../stores/appContext';
	import { bootstrapUser } from '$lib/bootstrap';

	let { children } = $props();

	onMount(() => {
		void bootstrapUser(appContext);
	});
</script>

<div class="preset-tonal-primary">
	{@render children?.()}
</div>
