/**
 * YouTube Shorts Watch History Exporter (COMP 4221 NLP Study)
 *
 * 100% Trusted-Types / CSP compliant (zero innerHTML sinks).
 * Uses pure document.createElement, textContent, and appendChild.
 */
(async function runShortsCollector() {
  if (!window.location.href.includes("youtube.com")) {
    alert("⚠️ Please open YouTube first!\n\nNavigate to https://www.youtube.com/feed/history and click this bookmark again.");
    window.location.href = "https://www.youtube.com/feed/history";
    return;
  }

  // Remove existing modal if already open
  const existingModal = document.getElementById("yt-shorts-collector-modal");
  if (existingModal) existingModal.remove();

  // Helper to create styled DOM elements without innerHTML (bypasses TrustedHTML)
  function createEl(tag, cssText, text = "") {
    const el = document.createElement(tag);
    if (cssText) el.style.cssText = cssText;
    if (text) el.textContent = text;
    return el;
  }

  // 1. Build Container
  const container = createEl(
    "div",
    "position:fixed;top:24px;right:24px;width:380px;max-width:calc(100vw - 48px);background:#0f0f13;color:#f1f1f1;border:1px solid #ff0033;border-radius:14px;box-shadow:0 16px 40px rgba(0,0,0,0.85);z-index:9999999;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;padding:20px;box-sizing:border-box;line-height:1.4;"
  );
  container.id = "yt-shorts-collector-modal";

  // Header row
  const header = createEl("div", "display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;");
  const titleBox = createEl("div", "font-weight:700;font-size:16px;color:#fff;display:flex;align-items:center;gap:8px;");
  const badge = createEl("span", "background:#ff0033;color:#fff;font-size:11px;font-weight:800;padding:2px 6px;border-radius:4px;", "SHORTS");
  const titleText = createEl("span", "", "History Exporter");
  titleBox.appendChild(badge);
  titleBox.appendChild(titleText);

  const closeBtn = createEl("button", "background:none;border:none;color:#888;font-size:18px;cursor:pointer;padding:0 4px;", "✕");
  closeBtn.onclick = () => container.remove();

  header.appendChild(titleBox);
  header.appendChild(closeBtn);
  container.appendChild(header);

  // Status text
  const statusEl = createEl("div", "font-size:13px;color:#aaa;margin-bottom:14px;", "Scanning your watch history...");
  container.appendChild(statusEl);

  // Progress bar
  const progressBar = createEl("div", "height:4px;background:#222;border-radius:2px;overflow:hidden;margin-bottom:14px;");
  const progressInner = createEl("div", "width:15%;height:100%;background:#ff0033;transition:width 0.3s ease;");
  progressBar.appendChild(progressInner);
  container.appendChild(progressBar);

  // Preview container
  const previewList = createEl("div", "max-height:140px;overflow-y:auto;background:#16161c;border-radius:8px;padding:10px;font-size:12px;color:#ccc;margin-bottom:14px;display:none;");
  container.appendChild(previewList);

  // Action buttons box
  const actionsBox = createEl("div", "display:none;flex-direction:column;gap:8px;");

  const downloadBtn = createEl("button", "background:#ff0033;color:#fff;border:none;padding:10px 14px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:6px;", "📥 Download my_shorts_history.json");
  const copyBtn = createEl("button", "background:#272733;color:#eee;border:1px solid #3c3c4f;padding:9px 14px;border-radius:8px;font-size:13px;font-weight:500;cursor:pointer;", "📋 Copy JSON to Clipboard");
  const noteEl = createEl("div", "font-size:11px;color:#777;text-align:center;margin-top:4px;", "COMP 4221 NLP Study • Anonymous & Private");

  actionsBox.appendChild(downloadBtn);
  actionsBox.appendChild(copyBtn);
  actionsBox.appendChild(noteEl);
  container.appendChild(actionsBox);

  document.body.appendChild(container);

  // 2. Smooth auto-scroll to fetch recent history
  const SCROLL_STEPS = 5;
  for (let i = 1; i <= SCROLL_STEPS; i++) {
    statusEl.textContent = `Loading history... scrolling (${i}/${SCROLL_STEPS})`;
    progressInner.style.width = `${(i / (SCROLL_STEPS + 1)) * 100}%`;
    window.scrollBy({ top: 1800, behavior: "smooth" });
    await new Promise((r) => setTimeout(r, 650));
  }

  statusEl.textContent = "Extracting YouTube Shorts...";
  progressInner.style.width = "95%";

  // 3. Extract shorts items from page
  const shortsList = [];
  const seenUrls = new Set();

  const shortAnchors = Array.from(document.querySelectorAll('a[href*="/shorts/"]'));

  for (const a of shortAnchors) {
    const rawHref = a.getAttribute("href") || "";
    const match = rawHref.match(/\/shorts\/([a-zA-Z0-9_-]+)/);
    if (!match) continue;

    const shortId = match[1];
    const canonicalUrl = `https://www.youtube.com/shorts/${shortId}`;
    if (seenUrls.has(canonicalUrl)) continue;

    // Look for video title
    let title = "";

    // 1. Try video title element inside closest renderer
    const renderer = a.closest("ytd-video-renderer, ytd-reel-item-renderer, ytd-rich-item-renderer, ytd-grid-video-renderer");
    if (renderer) {
      const titleEl = renderer.querySelector("#video-title, #title, h3");
      if (titleEl) {
        title = (titleEl.getAttribute("title") || titleEl.innerText || titleEl.textContent || "").trim();
      }
    }

    // 2. Try anchor's own attributes / text
    if (!title) {
      title = (a.getAttribute("title") || a.getAttribute("aria-label") || a.innerText || "").trim();
    }

    // 3. Fallback
    if (!title) {
      title = `YouTube Short (${shortId})`;
    }

    title = title.replace(/\s+by\s+.*$/i, "").replace(/\s+-\s+play\s+short$/i, "").trim();

    seenUrls.add(canonicalUrl);
    shortsList.push({ title, url: canonicalUrl });
  }

  progressInner.style.width = "100%";

  // 4. Present results in modal
  if (shortsList.length === 0) {
    statusEl.textContent = "⚠️ No Shorts found yet. Make sure you are on Watch History and have watched Shorts recently.";
    return;
  }

  statusEl.textContent = `✅ Successfully found ${shortsList.length} recent Shorts!`;

  // Render preview items
  previewList.style.display = "block";
  previewList.replaceChildren();

  shortsList.slice(0, 5).forEach((s, idx) => {
    const row = createEl("div", "margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;");
    const numSpan = createEl("span", "color:#ff0033;font-weight:bold;margin-right:4px;", `#${idx + 1}`);
    const labelSpan = createEl("span", "", s.title);
    row.appendChild(numSpan);
    row.appendChild(labelSpan);
    previewList.appendChild(row);
  });

  if (shortsList.length > 5) {
    const extra = createEl("div", "color:#666;font-size:11px;margin-top:4px;", `+ ${shortsList.length - 5} more items`);
    previewList.appendChild(extra);
  }

  actionsBox.style.display = "flex";

  const jsonString = JSON.stringify(shortsList, null, 2);

  // Download Button handler
  downloadBtn.onclick = () => {
    const blob = new Blob([jsonString], { type: "application/json" });
    const blobUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = blobUrl;
    a.download = "youtube_shorts_history.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(blobUrl);
    downloadBtn.textContent = "✅ Downloaded!";
    setTimeout(() => {
      downloadBtn.textContent = "📥 Download my_shorts_history.json";
    }, 2500);
  };

  // Copy Button handler
  copyBtn.onclick = async () => {
    try {
      await navigator.clipboard.writeText(jsonString);
      copyBtn.textContent = "✅ Copied to clipboard!";
    } catch (err) {
      const ta = document.createElement("textarea");
      ta.value = jsonString;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      copyBtn.textContent = "✅ Copied to clipboard!";
    }
    setTimeout(() => {
      copyBtn.textContent = "📋 Copy JSON to Clipboard";
    }, 2500);
  };
})();
