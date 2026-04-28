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

    const DIRECTIONS = ['down', 'up', 'right', 'left'];

    function _height(n) {
        if (!n.children || !n.children.length) return 0;
        return 1 + Math.max(...n.children.map(_height));
    }

    function assignLogical(n, leafCross) {
        n._fdepth = _height(n);
        if (!n.children || !n.children.length) {
            n._fcross = leafCross.get(n);
            return;
        }
        n.children.forEach(c => assignLogical(c, leafCross));
        n._fcross = n.children.reduce((s, c) => s + c._fcross, 0) / n.children.length;
    }

    function toScreen(node, dir, w, h, lead, trail, cross) {
        const H = Math.max(node._fdepth, 1);
        function _t(n) {
            const p = n._fdepth / H;
            const c = n._fcross;
            if (dir === 'down') {
                const span = h - lead - trail; const cspan = w - 2 * cross;
                n._fy = lead + p * span;
                n._fx = cross + c * cspan;
            } else if (dir === 'up') {
                const span = h - lead - trail; const cspan = w - 2 * cross;
                n._fy = (h - lead) - p * span;
                n._fx = cross + c * cspan;
            } else if (dir === 'right') {
                const span = w - lead - trail; const cspan = h - 2 * cross;
                n._fx = lead + p * span;
                n._fy = cross + c * cspan;
            } else if (dir === 'left') {
                const span = w - lead - trail; const cspan = h - 2 * cross;
                n._fx = (w - lead) - p * span;
                n._fy = cross + c * cspan;
            } else {
                throw new Error('unknown direction: ' + dir);
            }
            (n.children || []).forEach(_t);
        }
        _t(node);
    }

    function curvePath(cx, cy, px, py, dir, stroke, edgeWidth) {
        let d;
        if (dir === 'down' || dir === 'up') {
            const m = (cy + py) / 2;
            d = `M${cx.toFixed(1)},${cy.toFixed(1)} C${cx.toFixed(1)},${m.toFixed(1)} ${px.toFixed(1)},${m.toFixed(1)} ${px.toFixed(1)},${py.toFixed(1)}`;
        } else {
            const m = (cx + px) / 2;
            d = `M${cx.toFixed(1)},${cy.toFixed(1)} C${m.toFixed(1)},${cy.toFixed(1)} ${m.toFixed(1)},${py.toFixed(1)} ${px.toFixed(1)},${py.toFixed(1)}`;
        }
        return `<path d="${d}" stroke="${stroke}" stroke-width="${edgeWidth}" fill="none" stroke-linecap="round"/>`;
    }

    function outputPosition(dir, w, h, trail) {
        if (dir === 'down')  return [w / 2, h - trail * 0.35];
        if (dir === 'up')    return [w / 2, trail * 0.35];
        if (dir === 'right') return [w - trail * 0.35, h / 2];
        if (dir === 'left')  return [trail * 0.35, h / 2];
        return [w / 2, h - trail * 0.35];
    }

    function labelOffset(x, y, dir, fs, end) {
        const pad = 12;
        if (dir === 'down')  return [x, end === 'lead' ? y - pad : y + pad + fs];
        if (dir === 'up')    return [x, end === 'lead' ? y + pad + fs : y - pad];
        if (dir === 'right') return [end === 'lead' ? x - pad : x + pad, y + fs * 0.35];
        return [end === 'lead' ? x + pad : x - pad, y + fs * 0.35];
    }

    function textAnchor(dir, end) {
        if (dir === 'down' || dir === 'up') return 'middle';
        if (dir === 'right') return end === 'lead' ? 'end' : 'start';
        return end === 'lead' ? 'start' : 'end';
    }

    /**
     * Render an EML tree as an SVG flow diagram.
     *
     * @param {Array|Object} treeArr  Compact array form (preferred) or dict.
     * @param {Object} [opts]
     * @param {number} [opts.width=720]            Canvas width in px.
     * @param {number} [opts.height=420]           Canvas height in px.
     * @param {string} [opts.direction='down']     'down' | 'up' | 'right' | 'left'.
     * @param {string|Array} [opts.outputLabel='Out']  String, or list of strings for multi-valued formulas.
     * @param {boolean} [opts.expandSymbols=false] Replace named-symbol leaves with their EML construction.
     * @param {Array}  [opts.palette]
     * @param {number} [opts.labelFontSize=16]
     * @param {number} [opts.outputFontSize=20]
     * @param {number} [opts.edgeWidth=3]
     * @param {number} [opts.junctionRadius=4]
     */
    function renderFlowSvg(treeArr, opts) {
        opts = opts || {};
        const width          = opts.width          || 720;
        const height         = opts.height         || 420;
        const direction      = opts.direction      || 'down';
        const outputLabelArg = opts.outputLabel    || 'Out';
        const palette        = opts.palette        || DEFAULT_PALETTE;
        const labelFontSize  = opts.labelFontSize  || 16;
        const outputFontSize = opts.outputFontSize || 20;
        const edgeWidth      = opts.edgeWidth      || 3;
        const junctionRadius = opts.junctionRadius || 4;

        if (DIRECTIONS.indexOf(direction) < 0) {
            throw new Error('direction must be one of ' + DIRECTIONS.join(', '));
        }

        const isMulti = Array.isArray(outputLabelArg);
        const outputLabels = isMulti ? outputLabelArg.slice() : [outputLabelArg];

        let node = inflate(treeArr);
        node = binarize(node);

        const leaves = collectLeaves(node);
        if (!leaves.length) return '';
        const maxLabelLen = Math.max(...leaves.map(l => (l.label || '').length), 1);
        const halfLabelW  = 0.5 * 0.6 * labelFontSize * maxLabelLen;
        const crossMargin = Math.max(40, halfLabelW + 12);
        const leadMargin  = labelFontSize * 2.2;
        let trailMargin   = outputFontSize * 2.0;
        if (isMulti) trailMargin *= 1.6;

        const leafCross = new Map();
        if (leaves.length === 1) {
            leafCross.set(leaves[0], 0.5);
        } else {
            leaves.forEach((l, i) => leafCross.set(l, i / (leaves.length - 1)));
        }
        assignLogical(node, leafCross);
        toScreen(node, direction, width, height, leadMargin, trailMargin, crossMargin);

        const leafColor = new Map();
        leaves.forEach((l, i) => leafColor.set(l, palette[i % palette.length]));
        assignColors(node, leafColor);

        const parts = [`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" font-family="Inter, Helvetica, Arial, sans-serif">`];

        function emitEdges(n) {
            if (!n.children || !n.children.length) return;
            for (const c of n.children) {
                parts.push(curvePath(c._fx, c._fy, n._fx, n._fy, direction, rgbHex(c._fcolor), edgeWidth));
                emitEdges(c);
            }
        }
        emitEdges(node);

        const [outX, outY] = outputPosition(direction, width, height, trailMargin);
        if (!node.children || !node.children.length) {
            parts.push(curvePath(node._fx, node._fy, outX, outY, direction, rgbHex(node._fcolor), edgeWidth));
        }

        function emitJunctions(n) {
            if (!n.children || !n.children.length) return;
            parts.push(`<circle cx="${n._fx.toFixed(1)}" cy="${n._fy.toFixed(1)}" r="${junctionRadius}" fill="${rgbHex(n._fcolor)}" stroke="#222" stroke-width="0.8"/>`);
            (n.children || []).forEach(emitJunctions);
        }
        emitJunctions(node);

        // Leaf labels at the LEAD end.
        for (const l of leaves) {
            const [lx, ly] = labelOffset(l._fx, l._fy, direction, labelFontSize, 'lead');
            parts.push(`<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" fill="${rgbHex(l._fcolor)}" text-anchor="${textAnchor(direction, 'lead')}" font-weight="700" font-size="${labelFontSize}">${escSvg(l.label || '')}</text>`);
        }

        // Output label(s) at the TRAIL end.
        if (isMulti && outputLabels.length > 1) {
            const n = outputLabels.length;
            const spread = Math.max(60, outputFontSize * 1.2 * Math.max.apply(null, outputLabels.map(s => s.length)));
            outputLabels.forEach((lbl, i) => {
                const offset = (i - (n - 1) / 2) * spread;
                let ox = outX, oy = outY;
                if (direction === 'down' || direction === 'up') ox = outX + offset;
                else oy = outY + offset;
                parts.push(curvePath(node._fx, node._fy, ox, oy, direction, rgbHex(node._fcolor), edgeWidth));
                parts.push(`<text x="${ox.toFixed(1)}" y="${oy.toFixed(1)}" text-anchor="middle" font-weight="700" font-size="${outputFontSize}" fill="#222">${escSvg(lbl)}</text>`);
            });
            // ± indicator near the root junction
            let ix = node._fx, iy = node._fy;
            if (direction === 'down') iy += 16;
            else if (direction === 'up') iy -= 16;
            else if (direction === 'right') ix += 16;
            else ix -= 16;
            parts.push(`<text x="${ix.toFixed(1)}" y="${iy.toFixed(1)}" text-anchor="middle" font-size="${outputFontSize}" fill="#666" font-style="italic">±</text>`);
        } else {
            parts.push(`<text x="${outX.toFixed(1)}" y="${outY.toFixed(1)}" text-anchor="${textAnchor(direction, 'trail')}" font-weight="700" font-size="${outputFontSize}" fill="#222">${escSvg(outputLabels[0])}</text>`);
        }

        parts.push('</svg>');
        return parts.join('\n');
    }

    return {
        DEFAULT_PALETTE,
        DIRECTIONS,
        KIND_MAP,
        inflate,
        binarize,
        collectLeaves,
        renderFlowSvg,
    };
}));
