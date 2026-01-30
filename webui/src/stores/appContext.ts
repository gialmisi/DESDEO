import { browser } from '$app/environment';
import { writable } from 'svelte/store';
import type { UserPublic } from '$lib/gen/models';

export interface AppContextState {
	selectedProblemId: number | null;
	selectedMethod: string | null;
	selectedSessionId: number | null;
	selectedSessionInfo: string | null;
	user: UserPublic | null;
}

const storageKey = 'appContext';
const legacyStorageKey = 'methodSelection';

const defaultState: AppContextState = {
	selectedProblemId: null,
	selectedMethod: null,
	selectedSessionId: null,
	selectedSessionInfo: null,
	user: null
};

const loadInitialState = (): AppContextState => {
	if (!browser) {
		return defaultState;
	}

	const stored = localStorage.getItem(storageKey);
	if (stored) {
		return { ...defaultState, ...JSON.parse(stored) };
	}

	const legacy = localStorage.getItem(legacyStorageKey);
	if (legacy) {
		const legacyState = JSON.parse(legacy);
		const migrated = { ...defaultState, ...legacyState };
		localStorage.setItem(storageKey, JSON.stringify(migrated));
		localStorage.removeItem(legacyStorageKey);
		return migrated;
	}

	return defaultState;
};

const store = writable<AppContextState>(loadInitialState());

if (browser) {
	store.subscribe((value) => {
		localStorage.setItem(storageKey, JSON.stringify(value));
	});
}

function setProblem(problemId: number | null) {
	store.update((s) => ({
		...s,
		selectedProblemId: problemId,
		selectedMethod: null,
		selectedSessionId: null,
		selectedSessionInfo: null
	}));
}

function setMethod(method: string | null) {
	store.update((s) => ({
		...s,
		selectedMethod: method
	}));
}

function setSession(sessionId: number | null, sessionInfo: string | null = null) {
	store.update((s) => ({
		...s,
		selectedSessionId: sessionId,
		selectedSessionInfo: sessionId === null ? null : sessionInfo
	}));
}

function clearSession() {
	setSession(null, null);
}

function setUser(user: UserPublic | null) {
	store.update((s) => ({ ...s, user }));
}

function clearUser() {
	setUser(null);
}

function clearAll() {
	store.set(defaultState);
}

export const appContext = {
	subscribe: store.subscribe,
	setProblem,
	setMethod,
	setSession,
	clearSession,
	setUser,
	clearUser,
	clearAll,

	// pre-orval API (keep temporarily to avoid breakage)
	set: (problemId: number | null, method: string | null) => {
		store.set({
			...defaultState,
			selectedProblemId: problemId,
			selectedMethod: method
		});
	},
	clear: clearAll
};

export type AppContextStore = typeof appContext;
