import { get } from 'svelte/store';
import { appContext, type AppContextStore } from '$stores/appContext';
import { getCurrentUserInfoUserInfoGet } from '$lib/gen/endpoints/DESDEOFastAPI';
import { handleAuthFailure } from '$lib/handlers/auth';

let hasBootstrapped = false;
let bootstrapPromise: Promise<void> | null = null;

export async function bootstrapUser(
	store: AppContextStore = appContext,
	fetchImpl?: typeof fetch
): Promise<void> {
	if (hasBootstrapped) {
		return;
	}

	if (bootstrapPromise) {
		return bootstrapPromise;
	}

	bootstrapPromise = (async () => {
		const response = await getCurrentUserInfoUserInfoGet({ fetchImpl });

		if (response.status === 200) {
			store.setUser(response.data);
			hasBootstrapped = true;
			return;
		}

		if (!handleAuthFailure(response.status, store)) {
			store.setUser(null);
		}

		hasBootstrapped = true;
	})().finally(() => {
		bootstrapPromise = null;
	});

	return bootstrapPromise;
}

export function hasUserCached(store: AppContextStore = appContext): boolean {
	return Boolean(get(store).user);
}
