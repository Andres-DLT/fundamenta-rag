/*
 * Fundamenta en el navegador: búsqueda híbrida (BM25 + E5) sin servidor propio.
 *
 * Replica fundamenta/text.py, bm25.py y retriever.py sobre el índice que genera
 * scripts/build_web.py (index.json). La pregunta se codifica con el mismo modelo E5
 * cuantizado que se usó para los fragmentos, ejecutado con transformers.js.
 *
 * Si el índice indica una función de generación (api/answer.py), se le envían la
 * pregunta y los números de los fragmentos recuperados para obtener una respuesta
 * redactada con citas. Sin ella, la respuesta es extractiva, como en el servidor.
 *
 * Expone window.fundamentaEngine con la interfaz que usa fundamenta/static/index.html.
 */
(() => {
  const TRANSFORMERS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
  const PASSWORD_KEY = "fundamenta-password";

  let indexPromise;
  let extractorPromise;
  let password = readPassword();

  function readPassword() {
    try { return sessionStorage.getItem(PASSWORD_KEY); } catch { return null; }
  }

  function storePassword(value) {
    password = value;
    try {
      if (value) sessionStorage.setItem(PASSWORD_KEY, value);
      else sessionStorage.removeItem(PASSWORD_KEY);
    } catch { /* sin almacenamiento: se pedirá de nuevo al recargar */ }
  }

  function status(text) {
    window.dispatchEvent(new CustomEvent("fundamenta:status", { detail: text }));
  }

  function loadIndex() {
    indexPromise ??= fetch("index.json").then((response) => {
      if (!response.ok) throw new Error("No se pudo cargar el índice de documentos");
      return response.json();
    }).then((index) => {
      index.stopwords = new Set(index.text.stopwords);
      return index;
    });
    return indexPromise;
  }

  // --- Modelo de embeddings ------------------------------------------------------

  function loadExtractor(index) {
    extractorPromise ??= (async () => {
      const { pipeline, env } = await import(`${TRANSFORMERS}/dist/transformers.min.js`);
      env.allowLocalModels = false;
      const files = {};
      const extractor = await pipeline("feature-extraction", index.model.repo, {
        dtype: index.model.dtype,
        progress_callback: (event) => {
          if (event.status !== "progress" || !event.total) return;
          files[event.file] = event;
          const loaded = Object.values(files).reduce((sum, file) => sum + file.loaded, 0);
          const total = Object.values(files).reduce((sum, file) => sum + file.total, 0);
          status(`Preparando el modelo de búsqueda… ${Math.round((100 * loaded) / total)} %`);
        },
      });
      status(null);
      return extractor;
    })();
    extractorPromise.catch(() => { extractorPromise = undefined; });
    return extractorPromise;
  }

  async function embedQuery(index, question) {
    const extractor = await loadExtractor(index);
    const output = await extractor(index.model.prefix + question, { pooling: "mean", normalize: true });
    return output.data;
  }

  // --- Texto y BM25 (fundamenta/text.py y bm25.py) --------------------------------

  function lightStem(word, maxLength) {
    if (word.length > 3 && word.endsWith("s")) word = word.slice(0, -1);
    if (word.length > 4 && "aeo".includes(word.at(-1))) word = word.slice(0, -1);
    return word.slice(0, maxLength);
  }

  function tokenize(index, text) {
    const plain = text.toLowerCase().normalize("NFKD").replace(/\p{Mn}/gu, "");
    const words = plain.match(/[a-z0-9]+/g) || [];
    return words
      .filter((word) => !index.stopwords.has(word))
      .map((word) => lightStem(word, index.text.stem_max_length));
  }

  function lexicalRanking(index, question) {
    const { k1, b, average_length: averageLength, idf, lengths, term_frequencies: frequencies } = index.bm25;
    const terms = tokenize(index, question);
    const scores = frequencies.map((tf, i) => {
      const norm = k1 * (1 - b + (b * lengths[i]) / (averageLength || 1));
      let score = 0;
      for (const term of terms) {
        const freq = tf[term] || 0;
        if (freq) score += (idf[term] * freq * (k1 + 1)) / (freq + norm);
      }
      return score;
    });
    return scores
      .map((score, i) => [i, score])
      .sort((x, y) => y[1] - x[1])
      .filter(([, score]) => score > 0)
      .slice(0, index.retrieval.candidates);
  }

  // --- Búsqueda semántica e híbrida (fundamenta/retriever.py) -----------------------

  function semanticRanking(index, queryVector) {
    return index.vectors
      .map((vector, i) => [i, vector.reduce((sum, x, d) => sum + x * queryVector[d], 0)])
      .sort((x, y) => y[1] - x[1])
      .slice(0, index.retrieval.candidates);
  }

  function reciprocalRankFusion(rankings, k, weights) {
    const fused = new Map();
    rankings.forEach((ranking, r) => {
      ranking.forEach((i, position) => {
        fused.set(i, (fused.get(i) || 0) + weights[r] / (k + position + 1));
      });
    });
    return [...fused.entries()].sort((x, y) => y[1] - x[1]);
  }

  async function search(index, question) {
    const { k, rrf_k: rrfK, lexical_weight: lexicalWeight } = index.retrieval;
    const lexical = lexicalRanking(index, question).map(([i]) => i);
    const semantic = semanticRanking(index, await embedQuery(index, question)).map(([i]) => i);
    return reciprocalRankFusion([lexical, semantic], rrfK, [lexicalWeight, 1]).slice(0, k);
  }

  // --- Respuesta (fundamenta/generation.py) -----------------------------------------

  function label(chunk) {
    const parts = [chunk.source];
    if (chunk.section) parts.push(chunk.section.split(" > ").at(-1));
    if (chunk.page) parts.push(`página ${chunk.page}`);
    return parts.join(" › ");
  }

  function toSources(index, ranked) {
    return ranked.map(([i, score], position) => {
      const chunk = index.chunks[i];
      return { number: position + 1, ...chunk, score, label: label(chunk) };
    });
  }

  function extractive(question, sources) {
    const lines = ["Fragmentos más relevantes de la documentación:"];
    for (const source of sources) lines.push(`[${source.number}] ${source.label}\n${source.text}`);
    return { question, answer: lines.join("\n\n"), answered: true, mode: "extractive", sources };
  }

  async function callGeneration(index, body) {
    const response = await fetch(index.generation.url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Fundamenta-Password": password || "" },
      body: JSON.stringify({ ...body, fingerprint: index.fingerprint }),
    });
    if (response.status === 401) {
      storePassword(null);
      throw Object.assign(new Error("Contraseña incorrecta"), { unauthorized: true });
    }
    return response;
  }

  async function generate(index, question, ranked, sources) {
    const response = await callGeneration(index, { question, ids: ranked.map(([i]) => i) });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "El servicio de generación no está disponible");
    // La función no conoce las puntuaciones de la búsqueda: se toman de los resultados locales
    const scores = new Map(sources.map((source) => [source.number, source.score]));
    data.sources = data.sources.map((source) => ({ ...source, score: scores.get(source.number) ?? 0 }));
    return data;
  }

  // --- Interfaz pública --------------------------------------------------------------

  window.fundamentaEngine = {
    async health() {
      const index = await loadIndex();
      loadExtractor(index).catch(() => status("No se pudo cargar el modelo de búsqueda"));
      return {
        organization: index.organization,
        chunks: index.chunks.length,
        retrieval: "hybrid",
        generation: index.generation ? "ClaudeLLM" : "extractive",
        examples: index.examples,
      };
    },

    async requiresPassword() {
      const index = await loadIndex();
      return Boolean(index.generation) && !password;
    },

    async login(candidate) {
      const index = await loadIndex();
      password = candidate;
      let response;
      try {
        response = await callGeneration(index, { check: true });
      } catch (error) {
        if (error.unauthorized) return false;
        throw error;
      }
      if (!response.ok) throw new Error("El servicio de generación no está disponible");
      storePassword(candidate);
      return true;
    },

    async ask(question) {
      const index = await loadIndex();
      const ranked = await search(index, question);
      const sources = toSources(index, ranked);
      if (!index.generation) return extractive(question, sources);
      return generate(index, question, ranked, sources);
    },
  };
})();
