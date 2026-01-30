import { appContext, type AppContextStore } from '../../stores/appContext';

export function handleAuthFailure(status: number, store: AppContextStore = appContext): boolean {
	if (status === 401 || status === 403) {
		store.clearAll();
		return true;
	}
	return false;
}
