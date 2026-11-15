// Browser re-implementation of the sklearn TF-IDF (word 1-2 + char_wb 2-4, sublinear, l2) + multinomial LogReg.
const W = "[\\p{L}\\p{N}\\p{Mn}_]";
const TOKEN_RE = new RegExp(`(?<!${W})${W}+(?:'${W}+)?(?!${W})`, "gu");
export const normalize = (t) => String(t).replace(/\s+/g, " ").trim();

function wordGrams(text, [lo, hi]) {
  const toks = text.match(TOKEN_RE) || [], out = [];
  for (let n = lo; n <= hi; n++) for (let i = 0; i + n <= toks.length; i++) out.push(toks.slice(i, i + n).join(" "));
  return out;
}
function charWbGrams(text, [lo, hi]) {
  const out = [];
  for (const raw of text.replace(/\s\s+/g, " ").split(/\s+/).filter(Boolean)) {
    const w = Array.from(" " + raw + " ");
    for (let n = lo; n <= hi; n++) {
      let off = 0; out.push(w.slice(off, off + n).join(""));
      while (off + n < w.length) { off++; out.push(w.slice(off, off + n).join("")); }
      if (off === 0) break;
    }
  }
  return out;
}

export class TfidfModel {
  constructor(meta, buf) {
    const f = new Float32Array(buf); let o = 0;
    this.labels = meta.labels; this.F = meta.n_features; this.parts = [];
    let base = 0;
    for (const p of meta.parts) {
      const idx = new Map(p.vocab.map((v, i) => [v, i]));
      this.parts.push({ ...p, idx, base, idf: f.subarray(o, o + p.vocab.length) });
      o += p.vocab.length; base += p.vocab.length;
    }
    const K = this.labels.length;
    this.coef = f.subarray(o, o + K * this.F); o += K * this.F;
    this.b = f.subarray(o, o + K);
  }
  static async load(url = ".") {
    const [meta, buf] = await Promise.all([fetch(`${url}/tfidf_meta.json`).then(r => r.json()),
                                          fetch(`${url}/tfidf.bin`).then(r => r.arrayBuffer())]);
    return new TfidfModel(meta, buf);
  }
  features(text) {
    const s = normalize(text).toLowerCase(), feats = [];
    for (const p of this.parts) {
      const grams = p.analyzer === "char_wb" ? charWbGrams(s, p.ngram) : wordGrams(s, p.ngram);
      const cnt = new Map();
      for (const g of grams) { const j = p.idx.get(g); if (j !== undefined) cnt.set(j, (cnt.get(j) || 0) + 1); }
      const vals = []; let ss = 0;
      for (const [j, c] of cnt) { const v = (1 + Math.log(c)) * p.idf[j]; vals.push([p.base + j, v]); ss += v * v; }
      const nrm = Math.sqrt(ss) || 1;
      for (const [j, v] of vals) feats.push([j, v / nrm]);
    }
    return feats;
  }
  probs(text) {
    const x = this.features(text), K = this.labels.length, z = new Float64Array(K);
    for (let k = 0; k < K; k++) { let s = this.b[k]; const row = k * this.F; for (const [j, v] of x) s += this.coef[row + j] * v; z[k] = s; }
    const m = Math.max(...z); let t = 0;
    for (let k = 0; k < K; k++) { z[k] = Math.exp(z[k] - m); t += z[k]; }
    return Array.from(z, v => v / t);
  }
}
