/* ═══════════════════════════════════════════════════════════════
   VVDP Zalo Demo — Frontend Controller
═══════════════════════════════════════════════════════════════ */

// ── State ─────────────────────────────────────────────────────
let currentScenario = null;
let callActive      = false;
let callTimerInterval = null;
let callSeconds     = 0;
let chunkIndex      = 0;
let chunkTimer      = null;
let cumulativeText  = "";
let currentProb     = 0;
let warningShown    = false;
let isMuted         = false;
let isSpeaker       = false;
let scenariosData   = {};   // cache loaded from /api/scenario/<id>

const WARN_THRESHOLD = 0.45;   // show warning overlay above this probability

// ── DOM refs ──────────────────────────────────────────────────
const screens = {
  chat:     document.getElementById("screenChat"),
  incoming: document.getElementById("screenIncoming"),
  call:     document.getElementById("screenCall"),
};

// ── Clock ─────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  const h = String(now.getHours()).padStart(2, "0");
  const m = String(now.getMinutes()).padStart(2, "0");
  document.getElementById("sbTime").textContent = `${h}:${m}`;
}
setInterval(updateClock, 30000);
updateClock();

// ── Screen navigation ─────────────────────────────────────────
function showScreen(name) {
  Object.values(screens).forEach(s => s.classList.remove("active"));
  screens[name].classList.add("active");
}

// ── Scenario selection ────────────────────────────────────────
function selectScenario(id) {
  currentScenario = null;
  document.querySelectorAll(".scenario-card").forEach(c => c.classList.remove("selected"));
  const card = document.querySelector(`.scenario-card[data-id="${id}"]`);
  if (card) card.classList.add("selected");
  document.getElementById("btnSimulateCall").disabled = false;

  // Fetch full scenario data and cache it
  if (!scenariosData[id]) {
    fetch(`/api/scenario/${id}`)
      .then(r => r.json())
      .then(data => { scenariosData[id] = data; currentScenario = data; })
      .catch(e => console.error("Failed to load scenario", e));
  } else {
    currentScenario = scenariosData[id];
  }
}

// ── Incoming call simulation ──────────────────────────────────
function simulateIncomingCall() {
  if (!currentScenario) return;

  resetCallState();
  populateCallScreens(currentScenario);
  showScreen("incoming");

  // Auto-accept after 8 seconds (useful for batch testing)
  clearTimeout(window._autoAcceptTimer);
  window._autoAcceptTimer = setTimeout(() => {
    if (document.getElementById("screenIncoming").classList.contains("active")) {
      acceptCall();
    }
  }, 25000);
}

function populateCallScreens(s) {
  // Incoming screen
  const icAvatar = document.getElementById("icAvatar");
  icAvatar.textContent   = s.caller_avatar;
  icAvatar.style.background = s.caller_color;
  document.getElementById("icName").textContent  = s.caller_name;
  document.getElementById("icPhone").textContent = s.caller_phone;

  // Active call screen
  const callAvatar = document.getElementById("callAvatar");
  callAvatar.textContent      = s.caller_avatar;
  callAvatar.style.background = s.caller_color;
  document.getElementById("callName").textContent  = s.caller_name;
  document.getElementById("callPhone").textContent = s.caller_phone;

  // Call ended screen
  const ceoAvatar = document.getElementById("ceoAvatar");
  ceoAvatar.textContent      = s.caller_avatar;
  ceoAvatar.style.background = s.caller_color;
  document.getElementById("ceoName").textContent   = s.caller_name;
}

// ── Accept / Decline call ─────────────────────────────────────
function acceptCall() {
  clearTimeout(window._autoAcceptTimer);
  showScreen("call");
  document.getElementById("callStatus").textContent = "Đang kết nối...";
  document.getElementById("liveTranscript").innerHTML = '<div class="lt-waiting">⏳ Đang nghe...</div>';

  callActive  = true;
  callSeconds = 0;
  document.getElementById("callTimer").textContent = "00:00";
  callTimerInterval = setInterval(() => {
    callSeconds++;
    const m = String(Math.floor(callSeconds / 60)).padStart(2, "0");
    const s = String(callSeconds % 60).padStart(2, "0");
    document.getElementById("callTimer").textContent = `${m}:${s}`;
  }, 1000);

  // Start transcript simulation after 1.5s
  setTimeout(() => {
    document.getElementById("callStatus").textContent = "Đang gọi";
    if (currentScenario) startTranscriptPlayback(currentScenario.chunks);
  }, 1500);
}

function declineCall() {
  clearTimeout(window._autoAcceptTimer);
  showScreen("chat");
  resetCallState();
}

// ── Transcript playback ───────────────────────────────────────
function startTranscriptPlayback(chunks) {
  chunkIndex     = 0;
  cumulativeText = "";
  clearTranscriptUI();
  playNextChunk(chunks);
}

function playNextChunk(chunks) {
  if (!callActive || chunkIndex >= chunks.length) {
    if (callActive) {
      // Call naturally ended
      setTimeout(() => endCallNaturally(), 2000);
    }
    return;
  }

  const [speaker, text, delayMs] = chunks[chunkIndex];
  chunkTimer = setTimeout(() => {
    chunkIndex++;
    appendTranscriptLine(speaker, text);
    appendTranscriptLogLine(speaker, text);
    cumulativeText += " " + text;
    runDetection(text, cumulativeText.trim(), () => playNextChunk(chunks));
  }, delayMs);
}

// ── Detection ─────────────────────────────────────────────────
async function runDetection(text, cumulative, callback) {
  try {
    const resp = await fetch("/api/predict", {
      method:  "POST",
      headers: {"Content-Type": "application/json"},
      body:    JSON.stringify({ text, cumulative_text: cumulative }),
    });
    const result = await resp.json();
    updateDetectionUI(result);
    if (callback) callback();
  } catch (e) {
    console.error("Detection failed:", e);
    if (callback) callback();
  }
}

// ── Detection UI ──────────────────────────────────────────────
function updateDetectionUI(result) {
  const prob = result.probability;
  currentProb = prob;
  const pct = Math.round(prob * 100);

  // Testing panel stats
  document.getElementById("statProb").textContent  = `${pct}%`;
  document.getElementById("statLabel").textContent = result.label;
  document.getElementById("statConf").textContent  = result.confidence;
  document.getElementById("statMode").textContent  = result.mode === "model" ? "Model" : "Heuristic";

  colorStatVal("statProb",  prob >= WARN_THRESHOLD ? "#ef233c" : "#06d6a0");
  colorStatVal("statLabel", prob >= WARN_THRESHOLD ? "#ef233c" : "#06d6a0");

  // Probability meter (testing panel)
  document.getElementById("probMeterFill").style.width = `${pct}%`;

  // Live meter (call screen)
  document.getElementById("lmFill").style.width  = `${pct}%`;
  document.getElementById("lmPct").textContent   = `${pct}%`;

  // Signals
  const sigSection = document.getElementById("signalsSection");
  if (result.signals && result.signals.length > 0) {
    sigSection.style.display = "";
    const list = document.getElementById("signalsList");
    list.innerHTML = result.signals
      .map(s => `<span class="signal-tag">${s}</span>`)
      .join("");
  }

  // Detection banner on call screen
  if (prob >= WARN_THRESHOLD) {
    const banner = document.getElementById("detectionBanner");
    banner.classList.add("visible");
    document.getElementById("dbSub").textContent =
      `Xác suất: ${pct}% — ${result.confidence}`;
  }

  // Warning overlay (first time over threshold)
  if (prob >= WARN_THRESHOLD && !warningShown) {
    warningShown = true;
    showWarningOverlay(result);
  }
}

function colorStatVal(id, color) {
  document.getElementById(id).style.color = color;
}

// ── Transcript UI ─────────────────────────────────────────────
function clearTranscriptUI() {
  document.getElementById("liveTranscript").innerHTML = "";
}

function appendTranscriptLine(speaker, text) {
  const lt = document.getElementById("liveTranscript");
  const bubble = document.createElement("div");
  bubble.className = `lt-bubble ${currentProb >= WARN_THRESHOLD ? "scam-line" : ""}`;

  const spLabel = speaker === "caller"
    ? `<span class="speaker caller">Người gọi:</span>`
    : `<span class="speaker user">Bạn:</span>`;

  bubble.innerHTML = `${spLabel}<span class="msg">${escapeHtml(text)}</span>`;
  lt.appendChild(bubble);
  lt.scrollTop = lt.scrollHeight;
}

function appendTranscriptLogLine(speaker, text) {
  const log = document.getElementById("transcriptLog");
  const empty = log.querySelector(".tl-empty");
  if (empty) empty.remove();

  const line = document.createElement("div");
  line.className = "tl-line";
  const spLabel = speaker === "caller" ? "Người gọi" : "Bạn";
  const probPct  = Math.round(currentProb * 100);
  line.innerHTML = `
    <span class="tl-speaker ${speaker}">${spLabel}:</span>
    <span class="tl-text">${escapeHtml(text)}</span>
    <span class="tl-prob">${probPct}%</span>
  `;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}

// ── Warning overlay ───────────────────────────────────────────
function showWarningOverlay(result) {
  const pct = Math.round(result.probability * 100);
  document.getElementById("woProbVal").textContent = `${pct}%`;

  const body = document.getElementById("woBody");
  if (result.probability >= 0.7) {
    body.textContent = "Nguy hiểm rất cao! Đây là cuộc gọi lừa đảo (vishing). Đừng cung cấp bất kỳ thông tin cá nhân, mã OTP hay chuyển tiền theo yêu cầu!";
  } else {
    body.textContent = "Phát hiện dấu hiệu đáng ngờ trong cuộc gọi này. Hãy thận trọng và không cung cấp thông tin cá nhân hay chuyển tiền.";
  }

  const signalWrap = document.getElementById("woSignals");
  if (result.signals && result.signals.length > 0) {
    signalWrap.innerHTML = result.signals
      .map(s => `<span class="wo-signal">${s}</span>`)
      .join("");
    signalWrap.style.display = "flex";
  } else {
    signalWrap.style.display = "none";
  }

  document.getElementById("warningOverlay").classList.add("visible");
}

function closeWarning() {
  document.getElementById("warningOverlay").classList.remove("visible");
}

// ── Hang up ───────────────────────────────────────────────────
function hangUp() {
  clearTimeout(window._autoAcceptTimer);
  clearTimeout(chunkTimer);
  clearInterval(callTimerInterval);

  const duration = formatTime(callSeconds);
  const isScam   = currentScenario?.call_type === "scam";
  const detectedScam = currentProb >= WARN_THRESHOLD;

  document.getElementById("ceoDuration").textContent = `Thời gian: ${duration}`;
  const resultDiv = document.getElementById("ceoResult");
  if (isScam && detectedScam) {
    resultDiv.className = "ceo-result scam";
    resultDiv.innerHTML = "⚠️ Lừa đảo được phát hiện";
  } else if (!isScam && !detectedScam) {
    resultDiv.className = "ceo-result normal";
    resultDiv.innerHTML = "✅ Cuộc gọi bình thường";
  } else if (isScam && !detectedScam) {
    resultDiv.className = "ceo-result scam";
    resultDiv.innerHTML = "❌ Bỏ sót lừa đảo (False Negative)";
  } else {
    resultDiv.className = "ceo-result normal";
    resultDiv.innerHTML = "⚠️ Cảnh báo sai (False Positive)";
  }

  callActive = false;
  closeWarning();
  showScreen("chat");
  document.getElementById("callEndedOverlay").classList.add("visible");
}

function endCallNaturally() {
  hangUp();
}

function closeCallEnded() {
  document.getElementById("callEndedOverlay").classList.remove("visible");
  resetCallState();
}

// ── Call controls ─────────────────────────────────────────────
function toggleMute(btn) {
  isMuted = !isMuted;
  btn.classList.toggle("active", isMuted);
  btn.querySelector("span:first-child").textContent = isMuted ? "🔇" : "🎙️";
}

function toggleSpeaker(btn) {
  isSpeaker = !isSpeaker;
  btn.classList.toggle("active", isSpeaker);
}

// ── Batch test all scenarios ──────────────────────────────────
async function runAllScenarios() {
  const btn = document.getElementById("btnRunAll");
  btn.disabled = true;
  btn.textContent = "⏳ Đang chạy...";

  const batchSection = document.getElementById("batchResults");
  const tbody = document.getElementById("batchTableBody");
  batchSection.style.display = "";
  tbody.innerHTML = "";

  let correct = 0, total = 0;

  // Load all scenarios in parallel
  const ids = Array.from(document.querySelectorAll(".scenario-card")).map(c => c.dataset.id);
  await Promise.all(ids.map(id =>
    scenariosData[id] ? Promise.resolve() :
    fetch(`/api/scenario/${id}`).then(r => r.json()).then(d => { scenariosData[id] = d; })
  ));

  for (const id of ids) {
    const s = scenariosData[id];
    if (!s) continue;

    // Build full conversation text
    const fullText = s.chunks.map(([,text]) => text).join(" ");

    // Run detection on accumulated text
    const resp = await fetch("/api/predict", {
      method:  "POST",
      headers: {"Content-Type": "application/json"},
      body:    JSON.stringify({ text: fullText, cumulative_text: "" }),
    });
    const result = await resp.json();

    const predicted = result.label;
    const actual    = s.expected_label;
    const isCorrect = predicted === actual;

    if (isCorrect) correct++;
    total++;

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${s.caller_name}</td>
      <td style="color:${predicted === 'Lừa đảo' ? '#ef233c' : '#06d6a0'}">${predicted}</td>
      <td style="color:${actual === 'Lừa đảo' ? '#ef233c' : '#06d6a0'}">${actual}</td>
      <td class="${isCorrect ? 'correct' : 'wrong'}">${isCorrect ? '✅' : '❌'}</td>
      <td>${Math.round(result.probability * 100)}%</td>
    `;
    tbody.appendChild(tr);

    // Small delay between requests
    await sleep(200);
  }

  const accuracy = Math.round((correct / total) * 100);
  document.getElementById("batchSummary").textContent =
    `Độ chính xác: ${correct}/${total} = ${accuracy}%`;

  btn.disabled = false;
  btn.textContent = "▶️ Chạy tất cả kịch bản";
}

// ── Helpers ───────────────────────────────────────────────────
function resetCallState() {
  clearInterval(callTimerInterval);
  clearTimeout(chunkTimer);
  callActive    = false;
  callSeconds   = 0;
  chunkIndex    = 0;
  cumulativeText = "";
  currentProb   = 0;
  warningShown  = false;

  document.getElementById("detectionBanner").classList.remove("visible");
  document.getElementById("lmFill").style.width       = "0%";
  document.getElementById("lmPct").textContent        = "0%";
  document.getElementById("probMeterFill").style.width = "0%";
  document.getElementById("signalsSection").style.display = "none";
  document.getElementById("callTimer").textContent    = "00:00";
  document.getElementById("callStatus").textContent   = "Đang kết nối...";
  document.getElementById("statProb").textContent     = "—";
  document.getElementById("statLabel").textContent    = "—";
  document.getElementById("statConf").textContent     = "—";
  document.getElementById("statMode").textContent     = "—";
}

function formatTime(seconds) {
  const m = String(Math.floor(seconds / 60)).padStart(2, "0");
  const s = String(seconds % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Init ──────────────────────────────────────────────────────
(function init() {
  // Pre-load all scenario data
  document.querySelectorAll(".scenario-card").forEach(card => {
    const id = card.dataset.id;
    fetch(`/api/scenario/${id}`)
      .then(r => r.json())
      .then(d => { scenariosData[id] = d; })
      .catch(() => {});
  });
})();
