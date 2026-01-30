import createClient from 'openapi-fetch';
import type { paths } from './client-types';
import { browser } from '$app/environment';

const BASE_URL = import.meta.env.VITE_API_URL;

export const api = createClient<paths>({ baseUrl: BASE_URL });

export const serverApi = createClient<paths>({
	baseUrl: browser ? BASE_URL : (process.env.API_BASE_URL || 'http://localhost:8000')
});

api.use({
	async onRequest({ request }) {
		return new Request(request, { credentials: 'include' });
	},

	async onResponse({ request, response }) {
		if (response.status === 401) {
			const refreshRes = await fetch(`${BASE_URL}/refresh`, {
				method: 'POST',
				credentials: 'include'
			});

			if (refreshRes.ok) {
				const retryReq = new Request(request, { credentials: 'include' });
				return fetch(retryReq);
			}
		}

		return response;
	}
});
