/**
 * eml_flow.js — pure-JS port of eml_math.flow for in-browser rendering.
 *
 * Render a top-down flow diagram of an EML expression tree:
 *   - inputs labelled at the top, each in its own palette colour;
 *   - every internal junction is the binary primitive  eml(L, R) = exp(L) − ln(R)
 *     (no labels needed — all joins are the same operator);
 *   - junction colours are the average of the two child branch colours;
 *   - left branch is the exp-side input, right branch is the ln-side input;
 *   - output sits at the bottom, labelled with whatever name you pass.
 *
 * Trees are passed in the **compact** form produced by
 *   eml_math.tree.to_compact(node)
 * which is a JSON-friendly nested array:
 *   leaf     = [label, kindChar]
 *   internal = [label, kindChar, child1, child2, …]
 *
 * Usage
 * -----
 *   import {renderFlowSvg} from './eml_flow.js';     // ES module
 *   container.innerHTML = renderFlowSvg(treeArr, { outputLabel: "V_cb" });
 *
 * Or as a non-module script:
 *   <script src="eml_flow.js"></script>
 *   const svg = EmlFlow.renderFlowSvg(treeArr, { outputLabel: "V_cb" });
 */
(function (root, factory) {
    if (typeof exports === 'object' && typeof module !== 'undefined') {
        module.exports = factory();
    } else if (typeof define === 'function' && define.amd) {
        define([], factory);
    } else {
        root.EmlFlow = factory();
    }
}(typeof self !== 'undefined' ? self : this, function () {

    // ── Default palette (matches eml_math.flow.DEFAULT_PALETTE) ─────────
    const DEFAULT_PALETTE = [
        [231, 29, 54], [33, 196, 196], [31, 79, 231], [231, 33, 177],
        [33, 196, 79], [231, 131, 33], [131, 33, 231], [131, 79, 33],
        [33, 79, 131], [196, 231, 33], [196, 33, 131], [79, 131, 33],
    ];

    // ── kind-char ↔ kind-name (matches KIND_CHAR in eml_math.tree) ──────
    const KIND_MAP = {
        p: 'primitive', s: 'structural', c: 'compound', '#': 'scalar',
        v: 'vec',       P: 'pi',         C: 'const',    _: 'bottom',
        '?': 'unknown',
    };

    // ── Utilities ──────────────────────────────────────────────────────

    function inflate(treeArrOrDict) {
        if (!Array.isArray(treeArrOrDict)) return treeArrOrDict; // already inflated
        const [label, kc, ...kids] = treeArrOrDict;
        return {
            label: label,
            kind: KIND_MAP[kc] || 'unknown',
            children: kids.map(inflate),
        };
    }

    function rgbHex(rgb) {
        const c = v => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0');
        return '#' + c(rgb[0]) + c(rgb[1]) + c(rgb[2]);
    }

    function escSvg(s) {
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }

    // ── Tree transforms ────────────────────────────────────────────────

    /**
     * Make every internal node binary:
     *   - unary internal collapses (the child becomes the result);
     *   - n-ary (N>2) left-folds into nested binary.
     */
    function binarize(node) {
        if (!node.children || !node.children.length) return node;
        const cc = node.children.map(binarize);
        if (cc.length === 1) return cc[0];
        if (cc.length === 2) return { ...node, children: cc };
        let folded = cc[0];
        for (let i = 1; i < cc.length; i++) {
            folded = { ...node, children: [folded, cc[i]] };
        }
        return folded;
    }

    function collectLeaves(n) {
        if (!n.children || !n.children.length) return [n];
        const out = [];
        for (const c of n.children) out.push(...collectLeaves(c));
        return out;
    }

    function treeHeight(n) {
        if (!n.children || !n.children.length) return 0;
        return 1 + Math.max(...n.children.map(treeHeight));
    }

    // ── Layout ─────────────────────────────────────────────────────────

    function assignXY(n, leafX, top, layerH) {
        const h = treeHeight(n);
        n._fy = top + h * layerH;
        if (!n.children || !n.children.length) {
            n._fx = leafX.get(n);
            return;
        }
        n.children.forEach(c => assignXY(c, leafX, top, layerH));
        n._fx = n.children.reduce((s, c) => s + c._fx, 0) / n.children.length;
    }

    function assignColors(n, leafColor) {
        if (!n.children || !n.children.length) {
            n._fcolor = leafColor.get(n);
            return n._fcolor;
        }
        const cols = n.children.map(c => assignColors(c, leafColor));
        const avg = [
            cols.reduce((s, c) => s + c[0], 0) / cols.length,
            cols.reduce((s, c) => s + c[1], 0) / cols.length,
            cols.reduce((s, c) => s + c[2], 0) / cols.length,
        ];
        n._fcolor = avg;
        return avg;
    }

    // ── Public renderer ─────────────────────────────────────────────────

    /**
     * Render an EML tree as an SVG flow diagram.
     *
     * @param {Array|Object} treeArr  Compact array form (preferred) or {label,kind,children} dict.
     * @param {Object} [opts]
     * @param {number} [opts.width=720]            Canvas width in px.
     * @param {number} [opts.height=420]           Canvas height in px.
     * @param {string} [opts.outputLabel='Out']    Label drawn at the bottom.
     * @param {Array}  [opts.palette]              [[r,g,b], …]; defaults to DEFAULT_PALETTE.
     * @param {number} [opts.labelFontSize=16]     Pixel size for input labels.
     * @param {number} [opts.outputFontSize=20]    Pixel size for the output label.
     * @param {number} [opts.edgeWidth=3]          Stroke width of branch curves.
     * @param {number} [opts.junctionRadius=4]     Radius of the merge dots.
     * @returns {string} A self-contained <svg>…</svg> string.
     */
    function renderFlowSvg(treeArr, opts) {
        opts = opts || {};
        const width          = opts.width          || 720;
        const height         = opts.height         || 420;
        const outputLabel    = opts.outputLabel    || 'Out';
        const palette        = opts.palette        || DEFAULT_PALETTE;
        const labelFontSize  = opts.labelFontSize  || 16;
        const outputFontSize = opts.outputFontSize || 20;
        const edgeWidth      = opts.edgeWidth      || 3;
        const junctionRadius = opts.junctionRadius || 4;

        let node = inflate(treeArr);
        node = binarize(node);

        const leaves = collectLeaves(node);
        if (!leaves.length) return '';
        const maxLabelLen = Math.max(...leaves.map(l => (l.label || '').length), 1);
        const halfLabelW  = 0.5 * 0.6 * labelFontSize * maxLabelLen;
        const marginLR    = Math.max(40, halfLabelW + 12);
        const marginTop   = labelFontSize * 2.2;
        const marginBot   = outputFontSize * 2.0;

        const leafX = new Map();
        if (leaves.length === 1) {
            leafX.set(leaves[0], width / 2);
        } else {
            const usable = width - 2 * marginLR;
            leaves.forEach((l, i) => {
                leafX.set(l, marginLR + usable * i / (leaves.length - 1));
            });
        }
        const h = Math.max(treeHeight(node), 1);
        const layerH = (height - marginTop - marginBot) / h;
        assignXY(node, leafX, marginTop, layerH);

        const leafColor = new Map();
        leaves.forEach((l, i) => {
            leafColor.set(l, palette[i % palette.length]);
        });
        assignColors(node, leafColor);

        const parts = [`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" font-family="Inter, Helvetica, Arial, sans-serif">`];

        function emitEdges(n) {
            if (!n.children || !n.children.length) return;
            for (const c of n.children) {
                const my = (n._fy + c._fy) / 2;
                parts.push(`<path d="M${c._fx.toFixed(1)},${c._fy.toFixed(1)} C${c._fx.toFixed(1)},${my.toFixed(1)} ${n._fx.toFixed(1)},${my.toFixed(1)} ${n._fx.toFixed(1)},${n._fy.toFixed(1)}" stroke="${rgbHex(c._fcolor)}" stroke-width="${edgeWidth}" fill="none" stroke-linecap="round"/>`);
                emitEdges(c);
            }
        }
        emitEdges(node);

        // Synthetic edge from leaf to output position when binarisation
        // collapsed the entire tree to a single leaf.
        if (!node.children || !node.children.length) {
            const oy = height - marginBot;
            const my = (node._fy + oy) / 2;
            parts.push(`<path d="M${node._fx.toFixed(1)},${node._fy.toFixed(1)} C${node._fx.toFixed(1)},${my.toFixed(1)} ${(width/2).toFixed(1)},${my.toFixed(1)} ${(width/2).toFixed(1)},${oy.toFixed(1)}" stroke="${rgbHex(node._fcolor)}" stroke-width="${edgeWidth}" fill="none" stroke-linecap="round"/>`);
        }

        function emitJunctions(n) {
            if (!n.children || !n.children.length) return;
            parts.push(`<circle cx="${n._fx.toFixed(1)}" cy="${n._fy.toFixed(1)}" r="${junctionRadius}" fill="${rgbHex(n._fcolor)}" stroke="#222" stroke-width="0.8"/>`);
            for (const c of n.children) emitJunctions(c);
        }
        emitJunctions(node);

        for (const l of leaves) {
            parts.push(`<text x="${l._fx.toFixed(1)}" y="${(l._fy - 12).toFixed(1)}" fill="${rgbHex(l._fcolor)}" text-anchor="middle" font-weight="700" font-size="${labelFontSize}">${escSvg(l.label || '')}</text>`);
        }
        parts.push(`<text x="${(width/2).toFixed(1)}" y="${(height - marginBot * 0.35).toFixed(1)}" text-anchor="middle" font-weight="700" font-size="${outputFontSize}" fill="#222">${escSvg(outputLabel)}</text>`);
        parts.push('</svg>');
        return parts.join('\n');
    }

    return {
        DEFAULT_PALETTE,
        KIND_MAP,
        inflate,
        binarize,
        collectLeaves,
        renderFlowSvg,
    };
}));
