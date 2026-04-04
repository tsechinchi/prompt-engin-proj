const queryInput = document.getElementById("queryInput");
const modeSelect = document.getElementById("modeSelect");
const temperature = document.getElementById("temperature");
const fileUpload = document.getElementById("fileUpload");
const uploadList = document.getElementById("uploadList");
const askBtn = document.getElementById("askBtn");
const loadDemoBtn = document.getElementById("loadDemo");
const statusText = document.getElementById("statusText");
const answerBox = document.getElementById("answerBox");
const citationList = document.getElementById("citationList");
const tokenPill = document.getElementById("tokenPill");
const timetableSignal = document.getElementById("timetableSignal");
const newsSignal = document.getElementById("newsSignal");
const modeBadge = document.getElementById("modeBadge");
const tempReadout = document.getElementById("tempReadout");
const bleuBar = document.getElementById("bleuBar");
const rougeBar = document.getElementById("rougeBar");
const bleuValue = document.getElementById("bleuValue");
const rougeValue = document.getElementById("rougeValue");

const API_BASE = window.localStorage.getItem("hkbu_api_base") || "http://localhost:8000";
const USE_MOCK_CORPUS = false;

const STOP_WORDS = new Set([
  "the",
  "and",
  "for",
  "with",
  "this",
  "that",
  "from",
  "when",
  "what",
  "where",
  "how",
  "about",
  "into",
  "does",
  "is",
  "are",
  "can",
  "you",
]);

let uploadedDocs = [];

function tokensApprox(text) {
  return Math.max(1, Math.ceil(text.trim().split(/\s+/).length * 1.2));
}

function normalize(text) {
  return text.replace(/\s+/g, " ").trim();
}

function queryTerms(query) {
  return normalize(query)
    .toLowerCase()
    .split(" ")
    .filter((token) => token.length > 2 && !STOP_WORDS.has(token));
}

function shortExcerpt(text) {
  const cleaned = normalize(text);
  if (cleaned.length <= 220) {
    return cleaned;
  }
  return `${cleaned.slice(0, 220)}...`;
}

function splitIntoSnippets(text) {
  const normalized = normalize(text);
  if (!normalized) {
    return [];
  }

  const sentences = normalized.split(/(?<=[.!?])\s+/).filter(Boolean);
  if (sentences.length <= 1) {
    return [normalized];
  }

  const snippets = [];
  for (let i = 0; i < sentences.length; i += 2) {
    snippets.push(`${sentences[i]} ${sentences[i + 1] || ""}`.trim());
  }
  return snippets;
}

function topUploadSnippets(query, maxResults = 2) {
  const terms = queryTerms(query);
  if (!terms.length) {
    return [];
  }

  const ranked = [];

  for (const doc of uploadedDocs) {
    if (!doc.text) {
      continue;
    }

    for (const snippet of splitIntoSnippets(doc.text)) {
      const lowerSnippet = snippet.toLowerCase();
      let overlap = 0;
      for (const term of terms) {
        if (lowerSnippet.includes(term)) {
          overlap += 1;
        }
      }

      if (overlap > 0) {
        ranked.push({
          docName: doc.name,
          snippet,
          overlap,
        });
      }
    }
  }

  return ranked
    .sort((a, b) => b.overlap - a.overlap || b.snippet.length - a.snippet.length)
    .slice(0, maxResults);
}

function buildHelpfulUploadAnswer(query, snippets) {
  const primary = snippets[0];
  const secondary = snippets[1];
  const isOverviewQuery = /(lecture|overview|summary|about|topic)/i.test(query);

  if (isOverviewQuery) {
    if (secondary) {
      return `From ${primary.docName}, this lecture is mainly about: ${shortExcerpt(primary.snippet)} It also touches on: ${shortExcerpt(secondary.snippet)}`;
    }
    return `From ${primary.docName}, the most relevant explanation is: ${shortExcerpt(primary.snippet)}`;
  }

  if (secondary) {
    return `Best evidence from your uploads: ${shortExcerpt(primary.snippet)} Additional support: ${shortExcerpt(secondary.snippet)}`;
  }

  return `Best evidence from your uploads: ${shortExcerpt(primary.snippet)}`;
}

function updateLiveRadar(query = "", snippets = []) {
  if (!uploadedDocs.length) {
    timetableSignal.textContent =
      "No source loaded yet. Upload PDF/DOCX/PPTX or connect fetch_hkbu_updates() for live timetable/news.";
    newsSignal.textContent =
      "Live Radar shows source health and retrieval confidence (parsed files, overlap, and possible date cues).";
    return;
  }

  const parsedDocs = uploadedDocs.filter((doc) => doc.text);
  const parsedCount = parsedDocs.length;
  const totalKb = uploadedDocs.reduce((sum, doc) => sum + doc.sizeKb, 0);

  if (!parsedCount) {
    timetableSignal.textContent =
      "Files uploaded but no readable text extracted. Try another file or verify parser/CDN access.";
    newsSignal.textContent = `Upload payload detected: ${uploadedDocs.length} file(s), ${totalKb} KB.`;
    return;
  }

  const topSnippet = snippets[0]?.snippet || parsedDocs[0].text;
  const dateCue = topSnippet.match(
    /\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2})\b/i
  );

  if (dateCue) {
    timetableSignal.textContent = `Potential schedule cue found: "${dateCue[0]}". Verify against official HKBU timetable.`;
  } else {
    timetableSignal.textContent = `Parsed ${parsedCount} file(s) (${totalKb} KB). Ask about deadlines, dates, or timetable for sharper retrieval.`;
  }

  if (query) {
    const terms = queryTerms(query);
    const topLower = topSnippet.toLowerCase();
    const matchedTerms = terms.filter((term) => topLower.includes(term)).length;
    newsSignal.textContent = `Query overlap: ${matchedTerms}/${terms.length || 1} terms in top evidence snippet.`;
  } else {
    newsSignal.textContent = "Radar ready. Enter a question to see retrieval overlap and confidence hints.";
  }
}

function fileExtension(name) {
  const parts = name.toLowerCase().split(".");
  if (parts.length < 2) {
    return "";
  }
  return `.${parts.at(-1)}`;
}

function decodeXmlEntities(value) {
  return value.replace(/&(#x?[0-9a-fA-F]+|amp|lt|gt|quot|apos);/g, (match, token) => {
    const named = {
      amp: "&",
      lt: "<",
      gt: ">",
      quot: '"',
      apos: "'",
    };

    if (token in named) {
      return named[token];
    }

    if (token.startsWith("#x")) {
      const codePoint = Number.parseInt(token.slice(2), 16);
      return Number.isFinite(codePoint) ? String.fromCodePoint(codePoint) : match;
    }

    if (token.startsWith("#")) {
      const codePoint = Number.parseInt(token.slice(1), 10);
      return Number.isFinite(codePoint) ? String.fromCodePoint(codePoint) : match;
    }

    return match;
  });
}

async function parsePdfFile(file) {
  if (!window.pdfjsLib) {
    throw new Error("PDF parser library failed to load.");
  }

  if (!window.pdfjsLib.GlobalWorkerOptions.workerSrc) {
    window.pdfjsLib.GlobalWorkerOptions.workerSrc =
      "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.worker.min.js";
  }

  const arrayBuffer = await file.arrayBuffer();
  const pdf = await window.pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const pageTexts = [];

  for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
    const page = await pdf.getPage(pageNumber);
    const textContent = await page.getTextContent();
    const pageText = textContent.items
      .map((item) => ("str" in item ? item.str : ""))
      .join(" ");
    pageTexts.push(pageText);
  }

  return normalize(pageTexts.join(" "));
}

async function parseDocxFile(file) {
  if (!window.mammoth) {
    throw new Error("DOCX parser library failed to load.");
  }

  const arrayBuffer = await file.arrayBuffer();
  const result = await window.mammoth.extractRawText({ arrayBuffer });
  return normalize(result.value || "");
}

async function parsePptxFile(file) {
  if (!window.JSZip) {
    throw new Error("PPTX parser library failed to load.");
  }

  const arrayBuffer = await file.arrayBuffer();
  const zip = await window.JSZip.loadAsync(arrayBuffer);

  const slideNames = Object.keys(zip.files)
    .filter((name) => /^ppt\/slides\/slide\d+\.xml$/i.test(name))
    .sort((a, b) => {
      const aNum = Number.parseInt((a.match(/slide(\d+)\.xml/i) || ["", "0"])[1], 10);
      const bNum = Number.parseInt((b.match(/slide(\d+)\.xml/i) || ["", "0"])[1], 10);
      return aNum - bNum;
    });

  const slideTexts = [];
  for (const slideName of slideNames) {
    const xml = await zip.files[slideName].async("string");
    const matches = [...xml.matchAll(/<(?:a:)?t[^>]*>([\s\S]*?)<\/(?:a:)?t>/gi)];
    const plain = matches.map((match) => decodeXmlEntities(match[1])).join(" ");
    slideTexts.push(plain);
  }

  return normalize(slideTexts.join(" "));
}

function simulateAnswer(query, mode) {
  const uploadSnippets = topUploadSnippets(query, 2);
  if (uploadSnippets.length) {
    return {
      text: buildHelpfulUploadAnswer(query, uploadSnippets),
      citations: uploadSnippets.map(
        (item) => `Uploaded: ${item.docName} - ${shortExcerpt(item.snippet)}`
      ),
      bleu: 0.82,
      rouge: 0.86,
      radarSnippets: uploadSnippets,
    };
  }

  if (uploadedDocs.length && !uploadedDocs.some((doc) => doc.text)) {
    return {
      text: "I can see your uploaded files, but no readable text was extracted from PDF/DOCX/PPTX. Try another file or connect the upload flow to backend ingestion.",
      citations: uploadedDocs.map((doc) => `Uploaded: ${doc.name}`),
      bleu: 0.38,
      rouge: 0.41,
      radarSnippets: [],
    };
  }

  if (uploadedDocs.some((doc) => doc.text)) {
    return {
      text: "I parsed your uploads, but this query did not strongly match the extracted text. Try using keywords from your document title or a specific concept from the lecture slides.",
      citations: uploadedDocs.filter((doc) => doc.text).map((doc) => `Uploaded: ${doc.name}`),
      bleu: 0.4,
      rouge: 0.46,
      radarSnippets: [],
    };
  }

  const modeHint = {
    baseline: "Baseline mode gives a generic response without retrieval evidence.",
    bm25: "BM25 mode prioritizes lexical overlap from uploaded docs.",
    vector: "Vector mode prioritizes semantic similarity across chunks.",
    hybrid: "Hybrid mode fuses lexical and semantic scores for balance.",
  };

  return {
    text: `API fallback mode only. ${modeHint[mode]} Start backend with \"python run_api.py\" for graph-based answers, then re-run your query.`,
    citations: [],
    bleu: 0.3,
    rouge: 0.35,
    radarSnippets: [],
  };
}

function renderCitations(items) {
  citationList.innerHTML = "";
  if (!items.length) {
    const li = document.createElement("li");
    li.textContent = "No citations for this generic response.";
    citationList.appendChild(li);
    return;
  }

  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item;
    li.style.opacity = "0";
    li.style.transform = "translateY(6px)";
    citationList.appendChild(li);

    requestAnimationFrame(() => {
      li.style.transition = "opacity 220ms ease, transform 220ms ease";
      li.style.opacity = "1";
      li.style.transform = "translateY(0)";
    });
  }
}

function setMetric(value, barEl, textEl) {
  const bounded = Math.max(0, Math.min(1, value));
  barEl.value = bounded;
  textEl.textContent = bounded.toFixed(2);
}

function buildApiPayload(query) {
  return {
    query,
    mode: modeSelect.value,
    temperature: Number(temperature.value),
    top_k: 5,
    use_mock_generation: true,
    use_mock_corpus: USE_MOCK_CORPUS,
    uploaded_docs: uploadedDocs
      .filter((doc) => doc.text)
      .map((doc) => ({
        name: doc.name,
        text: doc.text,
      })),
  };
}

async function requestGraphAnswer(query) {
  const response = await fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(buildApiPayload(query)),
  });

  if (!response.ok) {
    throw new Error(`API request failed (${response.status})`);
  }

  const data = await response.json();
  return {
    text: data.answer || "No answer returned from API.",
    citations: Array.isArray(data.citations) ? data.citations : [],
    bleu: Number(data.quality?.bleu ?? 0.5),
    rouge: Number(data.quality?.rouge_l ?? 0.55),
    radarSnippets: Array.isArray(data.radar_snippets)
      ? data.radar_snippets.map((item) => ({ snippet: String(item.snippet || "") }))
      : [],
    tokenTotal: Number(data.tokens?.total_tokens ?? 0),
    status: String(data.status || "done"),
  };
}

function setStatus(text, mode = "ready") {
  statusText.textContent = text;
  statusText.classList.remove("status-idle");
  if (mode === "busy") {
    statusText.classList.add("status-idle");
  }
}

function refreshModeBadge() {
  const modeLabel = modeSelect.options[modeSelect.selectedIndex].text;
  modeBadge.textContent = `Mode: ${modeLabel}`;
}

function refreshTempReadout() {
  tempReadout.textContent = Number(temperature.value).toFixed(1);
}

async function parseUploadedFiles(fileList) {
  const parsed = [];
  for (const file of fileList) {
    const ext = fileExtension(file.name);
    let text = "";
    let readable = false;
    let parseStatus = "metadata only";
    let parseError = "";

    try {
      if (ext === ".pdf") {
        text = await parsePdfFile(file);
      } else if (ext === ".docx") {
        text = await parseDocxFile(file);
      } else if (ext === ".pptx") {
        text = await parsePptxFile(file);
      } else {
        parseStatus = "unsupported format";
      }

      if (text) {
        readable = true;
        parseStatus = "parsed";
      } else if (parseStatus === "metadata only") {
        parseStatus = "no text extracted";
      }
    } catch (error) {
      parseStatus = "parse failed";
      parseError = error instanceof Error ? error.message : "Unknown parser error";
    }

    parsed.push({
      name: file.name,
      sizeKb: Math.ceil(file.size / 1024),
      readable,
      parseStatus,
      parseError,
      text: readable ? normalize(text) : "",
    });
  }
  return parsed;
}

fileUpload.addEventListener("change", async () => {
  setStatus("Parsing uploads...", "busy");
  uploadedDocs = await parseUploadedFiles(fileUpload.files);
  uploadList.innerHTML = "";
  for (const file of uploadedDocs) {
    const li = document.createElement("li");
    const reason = file.parseError ? `, ${file.parseError}` : "";
    li.textContent = `${file.name} (${file.sizeKb} KB, ${file.parseStatus}${reason})`;
    uploadList.appendChild(li);
  }

  if (!uploadedDocs.length) {
    const li = document.createElement("li");
    li.textContent = "No files selected.";
    uploadList.appendChild(li);
    setStatus("Ready");
    return;
  }

  const parsedCount = uploadedDocs.filter((doc) => doc.readable).length;
  if (parsedCount > 0) {
    setStatus(`Parsed ${parsedCount} file(s)`);
  } else {
    setStatus("Uploads added, but no text extracted");
  }

  updateLiveRadar();
});

modeSelect.addEventListener("change", refreshModeBadge);
temperature.addEventListener("input", refreshTempReadout);

askBtn.addEventListener("click", async () => {
  const query = queryInput.value.trim();
  if (!query) {
    setStatus("Please enter a question.");
    return;
  }

  setStatus("Generating...", "busy");
  askBtn.disabled = true;

  try {
    const result = await requestGraphAnswer(query);
    const creativityPenalty = Number(temperature.value) * 0.06;

    answerBox.textContent = result.text;
    renderCitations(result.citations);

    if (result.tokenTotal > 0) {
      tokenPill.textContent = `Tokens: ${result.tokenTotal}`;
    } else {
      const outputTokens = tokensApprox(result.text);
      const inputTokens = tokensApprox(query);
      tokenPill.textContent = `Tokens: ${inputTokens + outputTokens}`;
    }

    setMetric(result.bleu - creativityPenalty, bleuBar, bleuValue);
    setMetric(result.rouge - creativityPenalty / 2, rougeBar, rougeValue);
    updateLiveRadar(query, result.radarSnippets || []);
    setStatus(result.status === "abstained" ? "Abstained" : "Done");
  } catch (_error) {
    // Fallback keeps the demo usable when backend is not running.
    const fallback = simulateAnswer(query, modeSelect.value);
    answerBox.textContent = fallback.text;
    renderCitations(fallback.citations);

    const outputTokens = tokensApprox(fallback.text);
    const inputTokens = tokensApprox(query);
    tokenPill.textContent = `Tokens: ${inputTokens + outputTokens}`;

    setMetric(fallback.bleu, bleuBar, bleuValue);
    setMetric(fallback.rouge, rougeBar, rougeValue);
    updateLiveRadar(query, fallback.radarSnippets || []);
    setStatus("API offline - using local demo");
  } finally {
    askBtn.disabled = false;
  }
});

loadDemoBtn.addEventListener("click", () => {
  queryInput.value = "When is the add/drop deadline for CS101?";
  modeSelect.value = "hybrid";
  temperature.value = "0.3";
  refreshModeBadge();
  refreshTempReadout();
  setStatus("Demo loaded");
});

refreshModeBadge();
refreshTempReadout();
updateLiveRadar();
