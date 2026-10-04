/**
 * A tiny observable store for app-wide state (model metadata, the current
 * route). View-local state such as the scrub position lives in the view.
 * Nothing here is persisted: patient data never touches browser storage.
 */

/**
 * Create a store.
 * @template T
 * @param {T} initial
 * @returns {{get: () => T, set: (patch: Partial<T>) => void,
 *   subscribe: (fn: (state: T) => void) => () => void}}
 */
export function createStore(initial) {
  let state = { ...initial };
  const listeners = new Set();
  return {
    get: () => state,
    set(patch) {
      state = { ...state, ...patch };
      for (const fn of listeners) fn(state);
    },
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}

/** The app's shared store. */
export const store = createStore({ meta: null, route: null });
