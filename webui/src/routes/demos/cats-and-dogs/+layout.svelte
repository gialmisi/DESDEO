<script lang="ts">
	import Dialogs from '$lib/components/custom/dialogs/Dialogs.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import '../../../app.css';
	let { children } = $props();
</script>

<main class="flex min-h-[calc(100vh-3rem)]">
	<Sidebar.Provider
		class="flex min-h-[calc(100vh-3rem)]"
		style="--sidebar-width: 20rem; --sidebar-width-mobile: 20rem;"
	>
		<Sidebar.Inset class="bg-gray-200">
			{@render children()}
		</Sidebar.Inset>
	</Sidebar.Provider>
</main>
<Dialogs />
