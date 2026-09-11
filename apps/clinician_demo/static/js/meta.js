/**
 * Lookups over the /api/meta payload: events, operating points, concepts.
 */

/**
 * The alert line for (event, horizon).
 * @param {object} meta
 * @param {string} event
 * @param {number} horizonHours
 * @returns {object|null}
 */
export function operatingPoint(meta, event, horizonHours) {
  return (
    meta.operating_points.find(
      (op) => op.event === event && Number(op.horizon_hours) === Number(horizonHours),
    ) ?? null
  );
}

/**
 * The event record by name.
 * @param {object} meta
 * @param {string} name
 * @returns {object}
 */
export function eventInfo(meta, name) {
  return meta.events.find((e) => e.name === name) ?? { name, display: name, short: name, definition: '' };
}

/**
 * Horizon key used by trace payloads: 24 -> "24h".
 * @param {number} hours
 * @returns {string}
 */
export function horizonKey(hours) {
  return `${Number(hours)}h`;
}
