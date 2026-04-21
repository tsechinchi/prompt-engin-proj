const queryInput = document.getElementById("queryInput");
const modeSelect = document.getElementById("modeSelect");
const compareModeSelect = document.getElementById("compareModeSelect");
const modeExplanation = document.getElementById("modeExplanation");
const temperature = document.getElementById("temperature");
const fileUpload = document.getElementById("fileUpload");
const uploadList = document.getElementById("uploadList");
const askBtn = document.getElementById("askBtn");
const loadDemoBtn = document.getElementById("loadDemo");
const statusText = document.getElementById("statusText");
const citationList = document.getElementById("citationList");
const tokenPill = document.getElementById("tokenPill");
const modelPill = document.getElementById("modelPill");
const compareSummary = document.getElementById("compareSummary");
const primaryModeTitle = document.getElementById("primaryModeTitle");
const comparisonModeTitle = document.getElementById("comparisonModeTitle");
const primaryModeSnippet = document.getElementById("primaryModeSnippet");
const comparisonModeSnippet = document.getElementById("comparisonModeSnippet");
const qualityComparison = document.getElementById("qualityComparison");
const tokenComparison = document.getElementById("tokenComparison");
const overallComparison = document.getElementById("overallComparison");
const modeBadge = document.getElementById("modeBadge");
const tempReadout = document.getElementById("tempReadout");
const conversationPane = document.getElementById("conversationPane");
const conversationStatus = document.getElementById("conversationStatus");
const clearConversationBtn = document.getElementById("clearConversationBtn");

const API_BASE = window.localStorage.getItem("hkbu_api_base") || "http://127.0.0.1:8000";
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

if (window.marked) {
  window.marked.setOptions({
    breaks: true,
    gfm: true
  });
}

const conversationHistory = [];

function normalizeChatText(text) {
  return String(text || "")
    .replace(/\r\n?/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function renderConversationMessage(message) {
  const wrapper = document.createElement("div");
  const roleClass = message.role === "user" ? "conversation-user" : "conversation-assistant";
  wrapper.className = `conversation-message ${roleClass}`;

  const label = document.createElement("div");
  label.className = "conversation-role";
  label.textContent = message.role === "user" ? "You" : "Assistant";

  const content = document.createElement("div");
  content.className = "conversation-content";
  const cleanedText = normalizeChatText(message.content);

  if (window.marked && message.role === "assistant") {
    content.innerHTML = marked.parse(cleanedText);
  } else {
    content.textContent = cleanedText;
  }

  wrapper.append(label, content);
  return wrapper;
}

function renderConversationPane() {
  if (!conversationPane) return;

  conversationPane.innerHTML = "";

  if (!conversationHistory.length) {
    const placeholder = document.createElement("p");
    placeholder.className = "conversation-empty";
    placeholder.textContent = "Start by asking a question to see the output here.";
    conversationPane.appendChild(placeholder);
    return;
  }

  for (const message of conversationHistory) {
    conversationPane.appendChild(renderConversationMessage(message));
  }
  conversationPane.scrollTop = conversationPane.scrollHeight;
}

function addConversationMessage(role, content) {
  conversationHistory.push({ role, content });
  renderConversationPane();
}

function updateLastAssistantMessage(content) {
  const lastMessage = conversationHistory[conversationHistory.length - 1];
  if (!lastMessage || lastMessage.role !== "assistant") {
    addConversationMessage("assistant", content);
    return;
  }
  lastMessage.content = content;
  renderConversationPane();
}

function clearConversationHistory() {
  conversationHistory.length = 0;
  renderConversationPane();
}

function setAssistantOutput(content) {
  updateLastAssistantMessage(content);
}

function tokensApprox(text) {
  return Math.max(1, Math.ceil(text.trim().split(/\s+/).length * 1.2));
}

function normalize(text) {
  // Preserve line breaks but collapse multiple blank lines
  return text
    .replace(/[ \t\r]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
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

  const sentences = normalized.split(/(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<!\bProf\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)(?<=[.!?])\s+/).filter(Boolean);
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
    baseline: "Non-RAG mode gives a generic response without retrieval evidence.",
    lexical: "RAG (Lexical) mode uses BM25 keyword retrieval to answer from provided context.",
    semantic: "RAG (Semantic) mode uses embedding-based retrieval to answer from provided context.",
    hybrid: "RAG mode uses both keyword and semantic retrieval to answer from provided context.",
  };

  return {
    text: `API fallback mode only. ${modeHint[mode] || modeHint.hybrid} Start backend with \"python run_api.py\" for graph-based answers, then re-run your query.`,
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

function setConversationStatus(text) {
  if (!conversationStatus) {
    return;
  }
  conversationStatus.textContent = text;
}

function getHistoryPayload() {
  const history = [...conversationHistory];

  if (history.length && history[history.length - 1]?.role === "assistant" && history[history.length - 1]?.content === "Thinking...") {
    history.pop();
  }

  if (history.length && history[history.length - 1]?.role === "user") {
    history.pop();
  }

  return history;
}

function buildApiPayload(query, mode) {
  return {
    query,
    mode,
    temperature: Number(temperature.value),
    use_mock_corpus: USE_MOCK_CORPUS,
    uploaded_docs: uploadedDocs
      .filter((doc) => doc.text)
      .map((doc) => ({
        name: doc.name,
        text: doc.text,
      })),
    history: getHistoryPayload(),
  };
}

async function requestGraphAnswer(query, mode) {
  const response = await fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(buildApiPayload(query, mode)),
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
    tokenTotal: Number(data.tokens?.total_tokens ?? 0),
    status: String(data.status || "done"),
    modelUsed: String(data.model_used || "unknown"),
  };
}

async function requestComparisonSummary(query, primaryMode, compareMode, primaryText, compareText) {
  if (!compareMode) {
    return {
      summary: "No comparison mode selected.",
    };
  }

  if (compareMode === primaryMode) {
    return {
      summary: `Primary and comparison mode are the same (${primaryMode}), so there is no difference to analyze.`,
    };
  }

  const response = await fetch(`${API_BASE}/api/compare`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      primary_mode: primaryMode,
      compare_mode: compareMode,
      primary_text: primaryText,
      compare_text: compareText,
      history: getHistoryPayload(),
    }),
  });

  if (!response.ok) {
    throw new Error(`Comparison request failed (${response.status})`);
  }

  const data = await response.json();
  return {
    summary: String(data.summary || "No comparison summary returned."),
  };
}

function updateModeComparison(primaryMode, primaryResult, compareMode, compareResult) {
  if (primaryModeTitle) {
    primaryModeTitle.textContent = `Primary mode: ${primaryMode}`;
  }

  if (comparisonModeTitle) {
    comparisonModeTitle.textContent = compareMode ? `Comparison mode: ${compareMode}` : "Comparison mode: none";
  }

  const primaryText = primaryResult?.text || "No output available.";
  const comparisonText = compareMode ? (compareResult?.text || "No comparison output available.") : "";

  if (primaryModeSnippet) {
    primaryModeSnippet.textContent = normalize(primaryText);
  }
  if (comparisonModeSnippet) {
    comparisonModeSnippet.textContent = normalize(comparisonText);
  }

  if (qualityComparison) {
    const primaryQuality = primaryResult ? averageQuality(primaryResult.bleu, primaryResult.rouge) : null;
    const compareQuality = compareMode && compareResult ? averageQuality(compareResult.bleu, compareResult.rouge) : null;

    if (!compareMode) {
      qualityComparison.textContent = "No comparison mode selected.";
    } else if (!compareResult) {
      qualityComparison.textContent = "Comparison not available for the selected mode.";
    } else {
      const better = compareQuality > primaryQuality ? "comparison" : compareQuality < primaryQuality ? "primary" : "both equally";
      qualityComparison.textContent = [
        `Primary quality: ${formatQuality(primaryQuality)} (BLEU ${formatNumber(primaryResult.bleu)}, ROUGE ${formatNumber(primaryResult.rouge)})`,
        `Comparison quality: ${formatQuality(compareQuality)} (BLEU ${formatNumber(compareResult.bleu)}, ROUGE ${formatNumber(compareResult.rouge)})`,
        `Better quality: ${better}`,
      ].join("\n");
    }
  }

  if (tokenComparison) {
    const primaryTokens = primaryResult?.tokenTotal ?? null;
    const compareTokens = compareMode && compareResult ? compareResult.tokenTotal : null;

    if (!compareMode) {
      tokenComparison.textContent = "No comparison mode selected.";
    } else if (!compareResult) {
      tokenComparison.textContent = "Comparison not available for the selected mode.";
    } else {
      const winner = compareTokens < primaryTokens ? "comparison" : compareTokens > primaryTokens ? "primary" : "both equal";
      tokenComparison.textContent = [
        `Primary tokens: ${primaryTokens}`,
        `Comparison tokens: ${compareTokens}`,
        `Better token usage: ${winner}`,
      ].join("\n");
    }
  }

  if (overallComparison) {
    if (!compareMode) {
      overallComparison.textContent = "No comparison mode selected.";
    } else if (!compareResult) {
      overallComparison.textContent = "Overall comparison not available for the selected mode.";
    } else {
      const primaryQuality = averageQuality(primaryResult.bleu, primaryResult.rouge);
      const compareQuality = averageQuality(compareResult.bleu, compareResult.rouge);
      const qualityDelta = compareQuality - primaryQuality;
      const tokenDelta = (primaryResult.tokenTotal ?? 0) - (compareResult.tokenTotal ?? 0);
      let overall = "primary";
      if (qualityDelta > 0.05 && tokenDelta > 0) {
        overall = "comparison";
      } else if (qualityDelta < -0.05 && tokenDelta < 0) {
        overall = "primary";
      } else if (Math.abs(qualityDelta) < 0.05 && Math.abs(tokenDelta) < 20) {
        overall = "too close to call";
      } else if (qualityDelta > 0.05) {
        overall = "comparison";
      } else if (tokenDelta > 20) {
        overall = "comparison";
      } else {
        overall = "primary";
      }
      overallComparison.textContent = `Overall better mode: ${overall}. Quality advantage: ${formatQuality(compareQuality - primaryQuality)}. Token delta: ${tokenDelta > 0 ? `comparison saved ${tokenDelta}` : tokenDelta < 0 ? `primary saved ${-tokenDelta}` : "equal"}.`;
    }
  }
}

function averageQuality(bleu, rouge) {
  return (Number(bleu) + Number(rouge)) / 2;
}

function formatQuality(value) {
  return value != null ? formatNumber(value) : "N/A";
}

function formatNumber(value) {
  return Number(value).toFixed(2);
}

function setStatus(text, mode = "ready") {
  statusText.textContent = text;
  statusText.classList.remove("status-idle");
  if (mode === "busy") {
    statusText.classList.add("status-idle");
  }
}

const MODE_EXPLANATIONS = {
  "baseline": "Baseline (no RAG): Generates an answer strictly from the model's internal base knowledge. Does not use retrieved documents.",
  "bm25": "BM25: Retrieves documents based strictly on keyword matching and frequency. Ideal for finding exact queries like names or acronyms.",
  "vector": "Vector: Retrieves documents based on semantic meaning using embeddings. Ideal for finding answers to conceptually phrased queries.",
  "hybrid": "Hybrid: Blends BM25 for precise keyword hits and Vector search for semantic understanding. It mixes the two for robust retrieval.",
  "thinking": "Thinking: AI reasons step by step with full transparency. Uses hybrid retrieval and streams the thinking process in real-time."
};

function refreshModeBadge() {
  const modeLabel = modeSelect.options[modeSelect.selectedIndex].text;
  modeBadge.textContent = `Mode: ${modeLabel}`;
  if (modeExplanation) {
    modeExplanation.textContent = MODE_EXPLANATIONS[modeSelect.value] || "";
  }
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
});

modeSelect.addEventListener("change", refreshModeBadge);
temperature.addEventListener("input", refreshTempReadout);

if (clearConversationBtn) {
  clearConversationBtn.addEventListener("click", () => {
    setConversationStatus("Live (Cleared)");
    setTimeout(() => setConversationStatus("Live"), 2000);

    clearConversationHistory();
    citationList.innerHTML = "";
    modelPill.textContent = "Model: None";
    modelPill.style.color = "inherit";
    modelPill.style.borderColor = "inherit";
    tokenPill.textContent = "Tokens: 0";

    compareSummary.textContent = "";
    if (primaryModeTitle) {
      primaryModeTitle.textContent = "Primary mode:";
    }
    if (comparisonModeTitle) {
      comparisonModeTitle.textContent = "Comparison mode: none";
    }
    if (primaryModeSnippet) {
      primaryModeSnippet.textContent = "";
    }
    if (comparisonModeSnippet) {
      comparisonModeSnippet.textContent = "";
    }
    if (qualityComparison) {
      qualityComparison.textContent = "";
    }
    if (tokenComparison) {
      tokenComparison.textContent = "";
    }
    if (overallComparison) {
      overallComparison.textContent = "";
    }

    // Clear query input
    queryInput.value = "";

    setStatus("Ready. Cleared all previous outputs.");
  });
}

askBtn.addEventListener("click", async () => {
  const query = queryInput.value.trim();
  if (!query) {
    setStatus("Please enter a question.");
    return;
  }

  addConversationMessage("user", query);
  addConversationMessage("assistant", "Thinking...");
  setConversationStatus("Waiting for response");

  setStatus("Ollama Generating...", "busy");
  askBtn.disabled = true;

  if (modeSelect.value === "thinking") {
    setStatus("🧠 Deep Thinking...", "busy");
    await requestThinkingAnswer(query);
    askBtn.disabled = false;
    return;
  }

    try {
    const primaryMode = modeSelect.value;
    const compareMode = compareModeSelect.value;
    const resultPromise = requestGraphAnswer(query, primaryMode);
    const comparisonPromise = compareMode && compareMode !== primaryMode
      ? requestGraphAnswer(query, compareMode)
      : Promise.resolve(null);

    const [result, comparisonResult] = await Promise.all([resultPromise, comparisonPromise]);
    const creativityPenalty = Number(temperature.value) * 0.06;

    updateLastAssistantMessage(result.text);
    renderCitations(result.citations);

    modelPill.textContent = `Model: ${result.modelUsed === "ollama" ? "Ollama" : String(result.modelUsed || "unknown")}`;
    if (result.modelUsed === "ollama") {
      modelPill.style.color = "var(--green-4)";
      modelPill.style.borderColor = "var(--green-4)";
    } else {
      modelPill.style.color = "inherit";
      modelPill.style.borderColor = "inherit";
    }

    if (result.tokenTotal > 0) {
      tokenPill.textContent = `Tokens: ${result.tokenTotal}`;
    } else {
      const outputTokens = tokensApprox(result.text);
      const inputTokens = tokensApprox(query);
      tokenPill.textContent = `Tokens: ${inputTokens + outputTokens}`;
    }

    updateModeComparison(primaryMode, result, compareMode, comparisonResult || { text: compareMode === primaryMode ? `Comparison mode is the same as primary (${compareMode}).` : "No comparison output available." });

    try {
      const summaryResult = await requestComparisonSummary(query, primaryMode, compareMode, result.text, comparisonResult?.text || "");
      if (compareSummary) {
        compareSummary.textContent = summaryResult.summary;
      }
    } catch (summaryError) {
      if (compareSummary) {
        compareSummary.textContent = "Comparison summary unavailable.";
      }
    }

    setConversationStatus("Response received");
    setStatus(result.status === "abstained" ? "Abstained" : "Done");
  } catch (_error) {
    // Fallback keeps the demo usable when backend is not running.
    const primaryMode = modeSelect.value;
    const compareMode = compareModeSelect.value;
    const fallback = simulateAnswer(query, primaryMode);
    const comparisonFallback = compareMode === primaryMode ? null : simulateAnswer(query, compareMode);

    updateLastAssistantMessage(fallback.text);
    renderCitations(fallback.citations);

    modelPill.textContent = "Model: Offline Demo";
    modelPill.style.color = "inherit";
    modelPill.style.borderColor = "inherit";

    const outputTokens = tokensApprox(fallback.text);
    const inputTokens = tokensApprox(query);
    tokenPill.textContent = `Tokens: ${inputTokens + outputTokens}`;

    updateModeComparison(primaryMode, fallback, compareMode, comparisonFallback || { text: compareMode === primaryMode ? `Comparison mode is the same as primary (${compareMode}).` : "No comparison output available." });
    if (compareSummary) {
      compareSummary.textContent = !compareMode
        ? "No comparison mode selected."
        : compareMode === primaryMode
          ? `Primary and comparison mode are the same (${compareMode}), so there is no difference to analyze.`
          : `Offline demo: only primary and comparison snippets are shown, no AI summary generated.`;
    }
    setConversationStatus("Offline demo response");
    setStatus("API offline - using local demo");
  } finally {
    askBtn.disabled = false;
  }
});

async function requestThinkingAnswer(query) {
  const thinkingSteps = [];
  let fullAnswer = "";
  let finalMeta = null;

  function buildLiveMessage() {
    let msg = "🧠 **Deep Thinking...**\n\n";
    for (const step of thinkingSteps) {
      msg += `✅ ${step}\n\n`;
    }
    if (!fullAnswer) {
      msg += "⏳ *Processing...*";
    } else {
      msg += "---\n\n" + fullAnswer;
    }
    return msg;
  }

  function buildFinalMessage() {
    const stepsHtml = thinkingSteps.map(s => `✅ ${s}`).join("<br>");
    return `<details>\n<summary>🧠 <strong>Thinking Process</strong> (${thinkingSteps.length} steps)</summary>\n<div class="thinking-steps-list">\n${stepsHtml}\n</div>\n</details>\n\n${fullAnswer}`;
  }

  try {
    const response = await fetch(`${API_BASE}/api/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildApiPayload(query)),
    });

    if (!response.ok) {
      throw new Error(`API request failed (${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop();

      for (const part of parts) {
        if (!part.trim()) continue;

        let eventType = "";
        let eventData = "";

        for (const line of part.split("\n")) {
          if (line.startsWith("event: ")) eventType = line.slice(7);
          else if (line.startsWith("data: ")) eventData += line.slice(6);
        }

        switch (eventType) {
          case "thinking_step":
            thinkingSteps.push(eventData.replace(/\\n/g, "\n"));
            updateLastAssistantMessage(buildLiveMessage());
            break;
          case "token":
            fullAnswer += eventData.replace(/\\n/g, "\n");
            updateLastAssistantMessage(buildLiveMessage());
            answerBox.innerHTML = window.marked ? marked.parse(fullAnswer) : fullAnswer;
            break;
          case "done":
            try { finalMeta = JSON.parse(eventData.replace(/\\n/g, "\n")); } catch (_e) { /* ignore */ }
            break;
          case "error":
            throw new Error(eventData);
        }
      }
    }

    // Finalize
    const finalMsg = buildFinalMessage();
    updateLastAssistantMessage(finalMsg);
    answerBox.innerHTML = window.marked ? marked.parse(fullAnswer) : fullAnswer;

    if (finalMeta) {
      renderCitations(finalMeta.citations || []);

      const modelStr = finalMeta.model_used === "ollama" ? "Ollama" : "Mock (Bypass/Fallback)";
      modelPill.textContent = `Model: ${modelStr}`;
      if (finalMeta.model_used === "ollama") {
        modelPill.style.color = "var(--green-4)";
        modelPill.style.borderColor = "var(--green-4)";
      } else {
        modelPill.style.color = "inherit";
        modelPill.style.borderColor = "inherit";
      }

      const totalTokens = finalMeta.tokens?.total_tokens || (tokensApprox(query) + tokensApprox(fullAnswer));
      tokenPill.textContent = `Tokens: ${totalTokens}`;

      const creativityPenalty = Number(temperature.value) * 0.06;
      setMetric((finalMeta.quality?.bleu || 0.5) - creativityPenalty, bleuBar, bleuValue);
      setMetric((finalMeta.quality?.rouge_l || 0.55) - creativityPenalty / 2, rougeBar, rougeValue);
    }

    updateLiveRadar(query, []);
    setConversationStatus("Response received");
    setStatus("Done (Thinking Mode)");

  } catch (_error) {
    const fallback = simulateAnswer(query, "hybrid");
    updateLastAssistantMessage(fallback.text);
    answerBox.innerHTML = window.marked ? marked.parse(fallback.text) : fallback.text;
    renderCitations(fallback.citations);

    modelPill.textContent = "Model: Offline Demo";
    modelPill.style.color = "inherit";
    modelPill.style.borderColor = "inherit";

    const outputTokens = tokensApprox(fallback.text);
    const inputTokens = tokensApprox(query);
    tokenPill.textContent = `Tokens: ${inputTokens + outputTokens}`;

    setMetric(fallback.bleu, bleuBar, bleuValue);
    setMetric(fallback.rouge, rougeBar, rougeValue);
    updateLiveRadar(query, []);
    setConversationStatus("Offline demo response");
    setStatus("API offline - using local demo");
  }
}

loadDemoBtn.addEventListener("click", () => {
  queryInput.value = "When is the add/drop deadline for CS101?";
  modeSelect.value = "hybrid";
  compareModeSelect.value = "baseline";
  temperature.value = "0.3";
  refreshModeBadge();
  refreshTempReadout();
  setStatus("Demo loaded");
});

refreshModeBadge();
refreshTempReadout();
renderConversationPane();
