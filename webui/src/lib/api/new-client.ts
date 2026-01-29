// NOTE: Supports cases where `content-type` is other than `json`
const getBody = async <T>(c: Response | Request): Promise<T> => {
  // If it's a Response and there is explicitly no content, don't parse anything
  if (c instanceof Response && (c.status === 204 || c.status === 205)) {
    return null as T;
  }

  const contentType = c.headers.get('content-type');

  if (contentType && contentType.includes('application/json')) {
    // Avoid JSON.parse errors on empty bodies
    const text = await (c as Response).text?.() ?? '';
    if (!text) return null as T;
    return JSON.parse(text) as T;
  }

  if (contentType && contentType.includes('application/pdf')) {
    return (c as Response).blob() as Promise<T>;
  }

  return (c as Response).text() as Promise<T>;
};

const resolveBaseUrl = (): string => {
  const isBrowser = typeof window !== 'undefined';
  const envBase =
    (typeof process !== 'undefined' && process.env?.API_BASE_URL) ||
    (typeof process !== 'undefined' && process.env?.VITE_API_URL) ||
    '';
  const viteBase =
    typeof import.meta !== 'undefined' ? import.meta.env?.VITE_API_URL : '';

  if (isBrowser) {
    const origin = window.location.origin;
    const base = viteBase || '/api';
    if (!base) {
      return origin;
    }
    if (base.startsWith('http')) {
      return base;
    }
    return `${origin}${base}`;
  }

  if (envBase) {
    return envBase;
  }

  return 'http://localhost:8000';
};

// NOTE: Update just base url
const getUrl = (contextUrl: string): string => {
  if (contextUrl.startsWith('http')) {
    return contextUrl;
  }

  const base = resolveBaseUrl();
  const normalized = contextUrl.replace(/^undefined/, '');
  const path = normalized.startsWith('/') ? normalized : `/${normalized}`;

  return new URL(path, base).toString();
};

const getHeaders = (headers?: HeadersInit): HeadersInit => {
  return {
    ...headers,
    // add headers if needed
  }
};

export const customFetch = async <T>(
  url: string,
  options: (RequestInit & {fetchImpl?: typeof fetch }),
): Promise<T> => {
  const f = options.fetchImpl ?? fetch;

  const requestUrl = getUrl(url);
  const requestHeaders = getHeaders(options.headers);

  const requestInit: RequestInit = {
    ...options,
    headers: requestHeaders,
    credentials: "include",
  };

  const response = await f(requestUrl, requestInit);
  const data = await getBody<T>(response);

  return { status: response.status, data, headers: response.headers } as T;
};
