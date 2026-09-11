/**
 * The only module that talks to the server. Every request carries the
 * demo header the server requires, so a page on another origin cannot
 * drive the API through the viewer's browser.
 */

const DEMO_HEADER = { 'X-Odyssey-Demo': '1' };

/** An API failure carrying the server's message and HTTP status. */
export class ApiError extends Error {
  /**
   * @param {string} message
   * @param {number} status
   */
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/**
 * Send one JSON request.
 * @param {string} path
 * @param {{method?: string, body?: *}} [opts]
 * @returns {Promise<*>} the parsed JSON body
 */
export async function request(path, { method = 'GET', body } = {}) {
  const headers = { ...DEMO_HEADER, Accept: 'application/json' };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  let response;
  try {
    response = await fetch(path, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      cache: 'no-store',
      credentials: 'same-origin',
    });
  } catch {
    throw new ApiError('Cannot reach the demo server. Is the SSH tunnel still open?', 0);
  }
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const message = payload && typeof payload.error === 'string'
      ? payload.error
      : `The server answered ${response.status}.`;
    throw new ApiError(message, response.status);
  }
  return payload;
}

const enc = encodeURIComponent;

/** Typed shortcuts for every endpoint the UI uses. */
export const api = {
  meta: () => request('/api/meta'),
  gallery: () => request('/api/gallery'),
  patient: (sid) => request(`/api/patients/${enc(sid)}`),
  trace: (sid, vid) => request(`/api/patients/${enc(sid)}/visits/${enc(vid)}/trace`),
  presets: () => request('/api/whatif/presets'),
  whatif: (sid, vid, body) =>
    request(`/api/patients/${enc(sid)}/visits/${enc(vid)}/whatif`, { method: 'POST', body }),
  evidence: (sid, vid, body) =>
    request(`/api/patients/${enc(sid)}/visits/${enc(vid)}/evidence`, { method: 'POST', body }),
  job: (jobId) => request(`/api/jobs/${enc(jobId)}`),
  scorecard: () => request('/api/scorecard'),
};
