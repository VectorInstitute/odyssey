/**
 * Small DOM helpers. Everything is built with createElement and CSSOM
 * writes, never with inline style/handler attributes, so the page works
 * under a strict Content-Security-Policy.
 */

const SVG_NS = 'http://www.w3.org/2000/svg';

function appendChildren(node, children) {
  for (const child of [children].flat(Infinity)) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

/**
 * Create an HTML element.
 * @param {string} tag
 * @param {Object<string, *>} [props] class, text, dataset, style (object of
 *   CSS properties), on* listeners, or plain attributes.
 * @param {Array|Node|string} [children]
 * @returns {HTMLElement}
 */
export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key === 'style') {
      for (const [prop, v] of Object.entries(value)) node.style.setProperty(prop, v);
    } else if (key.startsWith('on') && typeof value === 'function') {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else node.setAttribute(key, value === true ? '' : String(value));
  }
  appendChildren(node, children);
  return node;
}

/**
 * Create an SVG element with attributes.
 * @param {string} tag
 * @param {Object<string, *>} [attrs]
 * @param {Array|Node|string} [children]
 * @returns {SVGElement}
 */
export function svg(tag, attrs = {}, children = []) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key === 'text') node.textContent = value;
    else if (key === 'class') node.setAttribute('class', value);
    else node.setAttribute(key, String(value));
  }
  appendChildren(node, children);
  return node;
}

/**
 * A centered "working on it" message.
 * @param {string} message
 * @returns {HTMLElement}
 */
export function loadingBlock(message) {
  return el('div', { class: 'state-msg', role: 'status' }, [el('div', { class: 'spinner' }), message]);
}

/**
 * A centered error message.
 * @param {Error|string} err
 * @returns {HTMLElement}
 */
export function errorBlock(err) {
  const text = err instanceof Error ? err.message : String(err);
  return el('div', { class: 'state-msg state-msg--error', role: 'alert' }, text || 'Something went wrong.');
}

/**
 * A centered "nothing to show" message.
 * @param {string} message
 * @returns {HTMLElement}
 */
export function emptyBlock(message) {
  return el('div', { class: 'state-msg' }, message);
}

/**
 * A titled card.
 * @param {string|Node} title
 * @param {Array|Node} children
 * @param {{sub?: string, actions?: Array|Node, className?: string}} [opts]
 * @returns {HTMLElement}
 */
export function card(title, children, opts = {}) {
  const head = el('div', { class: 'card__head' }, [
    typeof title === 'string' ? el('h2', { text: title }) : title,
    opts.sub ? el('span', { class: 'card__sub', text: opts.sub }) : null,
    opts.actions ? el('span', { class: 'spacer' }) : null,
    opts.actions ?? null,
  ]);
  return el('section', { class: `card ${opts.className ?? ''}`.trim() }, [head, children]);
}
