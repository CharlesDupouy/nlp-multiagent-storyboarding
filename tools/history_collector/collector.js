/**
 * YouTube Shorts Watch History Exporter (COMP 4221 NLP Study)
 *
 * 100% Trusted-Types and CSP compliant:
 * Uses pure document.createElement, textContent, and appendChild (zero innerHTML sinks).
 */
(async function runShortsCollector() {
  if (!window.location.href.includes("youtube.com")) {
    alert("Please open YouTube first.\n\nNavigate to https://www.youtube.com/feed/history and click this bookmark again.");
    window.location.href = "https://www.youtube.com/feed/history";
    return;
  }

  // Remove existing modal if already open
  const existingModal = document.getElementById("yt-shorts-collector-modal");
  if (existingModal) existingModal.remove();

  // Helper to create styled DOM elements without innerHTML (Trusted Types safe)
  function createEl(tag, cssText, text = "") {
    const el = document.createElement(tag);
    if (cssText) el.style.cssText = cssText;
    if (text) el.textContent = text;
    return el;
  }

  // 1. Root Container
  const container = createEl(
    "div",
    "position:fixed;top:24px;right:24px;width:380px;max-width:calc(100vw - 48px);background:#10131b;color:#f1f5f9;border:1px solid rgba(255,255,255,0.12);border-radius:12px;box-shadow:0 12px 36px rgba(0,0,0,0.65);z-index:9999999;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;padding:18px;box-sizing:border-box;line-height:1.45;"
  );
  container.id = "yt-shorts-collector-modal";

  // Header row
  const header = createEl("div", "display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;");
  const titleBox = createEl("div", "font-weight:600;font-size:14px;color:#f8fafc;display:flex;align-items:center;gap:8px;");
  const badge = createEl("span", "background:#e50914;color:#fff;font-size:10px;font-weight:700;letter-spacing:0.5px;padding:2px 6px;border-radius:4px;text-transform:uppercase;", "Shorts");
  const titleText = createEl("span", "", "History Exporter");
  titleBox.appendChild(badge);
  titleBox.appendChild(titleText);

  const closeBtn = createEl("button", "background:none;border:none;color:#94a3b8;font-size:16px;cursor:pointer;padding:2px 6px;border-radius:4px;line-height:1;", "✕");
  closeBtn.title = "Close (Esc)";
  closeBtn.onmouseenter = () => { closeBtn.style.color = "#fff"; };
  closeBtn.onmouseleave = () => { closeBtn.style.color = "#94a3b8"; };
  closeBtn.onclick = () => container.remove();

  // Escape key listener to close modal
  const handleKeyDown = (e) => {
    if (e.key === "Escape") {
      container.remove();
      window.removeEventListener("keydown", handleKeyDown);
    }
  };
  window.addEventListener("keydown", handleKeyDown);

  header.appendChild(titleBox);
  header.appendChild(closeBtn);
  container.appendChild(header);

  // Status message
  const statusEl = createEl("div", "font-size:13px;color:#94a3b8;margin-bottom:12px;", "Scanning watch history...");
  container.appendChild(statusEl);

  // Progress Bar
  const progressBar = createEl("div", "height:4px;background:#1e293b;border-radius:2px;overflow:hidden;margin-bottom:14px;");
  const progressInner = createEl("div", "width:15%;height:100%;background:#e50914;transition:width 0.25s ease;");
  progressBar.appendChild(progressInner);
  container.appendChild(progressBar);

  // Preview List Container
  const previewList = createEl("div", "max-height:140px;overflow-y:auto;background:#090b10;border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:10px 12px;font-size:12px;color:#cbd5e1;margin-bottom:14px;display:none;");
  container.appendChild(previewList);

  // Actions Container
  const actionsBox = createEl("div", "display:none;flex-direction:column;gap:8px;");

  const downloadBtn = createEl("button", "background:#e50914;color:#fff;border:none;padding:9px 14px;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:6px;transition:opacity 0.15s ease;", "Download JSON");
  downloadBtn.onmouseenter = () => { downloadBtn.style.opacity = "0.9"; };
  downloadBtn.onmouseleave = () => { downloadBtn.style.opacity = "1"; };

  const copyBtn = createEl("button", "background:#1e293b;color:#f1f5f9;border:1px solid rgba(255,255,255,0.08);padding:8px 14px;border-radius:6px;font-size:13px;font-weight:500;cursor:pointer;transition:background 0.15s ease;", "Copy JSON to Clipboard");
  copyBtn.onmouseenter = () => { copyBtn.style.background = "#283548"; };
  copyBtn.onmouseleave = () => { copyBtn.style.background = "#1e293b"; };

  const noteEl = createEl("div", "font-size:11px;color:#64748b;text-align:center;margin-top:2px;", "HKUST COMP 4221 Study • Anonymous & Client-Side");

  actionsBox.appendChild(downloadBtn);
  actionsBox.appendChild(copyBtn);
  actionsBox.appendChild(noteEl);
  container.appendChild(actionsBox);

  document.body.appendChild(container);

  // 2. Smooth auto-scroll down the history feed
  const SCROLL_STEPS = 5;
  for (let i = 1; i <= SCROLL_STEPS; i++) {
    statusEl.textContent = `Loading history... scrolling (${i}/${SCROLL_STEPS})`;
    progressInner.style.width = `${(i / (SCROLL_STEPS + 1)) * 100}%`;
    window.scrollBy({ top: 1800, behavior: "smooth" });
    await new Promise((r) => setTimeout(r, 650));
  }

  statusEl.textContent = "Extracting Shorts...";
  progressInner.style.width = "95%";

  // 3. Collect Shorts elements
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

    let title = "";

    // A. Check renderer container
    const renderer = a.closest("ytd-reel-item-renderer, ytd-video-renderer, ytd-rich-item-renderer, ytd-grid-video-renderer, ytd-item-section-renderer");
    if (renderer) {
      const titleEl = renderer.querySelector("#headline, #video-title, #video-title-link, .reel-item-endpoint span, .reel-item-endpoint, h3, #title, yt-formatted-string");
      if (titleEl) {
        title = (titleEl.getAttribute("title") || titleEl.innerText || titleEl.textContent || "").trim();
      }
      if (!title) {
        const thumbImg = renderer.querySelector("img#img[alt]");
        if (thumbImg && thumbImg.getAttribute("alt")) {
          title = thumbImg.getAttribute("alt").trim();
        }
      }
    }

    // B. Check anchor itself
    if (!title) {
      title = (a.getAttribute("title") || a.getAttribute("aria-label") || a.innerText || "").trim();
    }

    // C. Fallback
    if (!title) {
      title = `YouTube Short (${shortId})`;
    }

    title = title.replace(/\s+by\s+.*$/i, "").replace(/\s+-\s+play\s+short$/i, "").trim();

    seenUrls.add(canonicalUrl);
    shortsList.push({ title, url: canonicalUrl });
  }

  // 4. Enrich missing titles via public YouTube oEmbed
  const missingItems = shortsList.filter((s) => s.title.startsWith("YouTube Short (") || !s.title);
  if (missingItems.length > 0) {
    statusEl.textContent = `Resolving titles (${missingItems.length} videos)...`;
    for (let i = 0; i < missingItems.length; i += 5) {
      const batch = missingItems.slice(i, i + 5);
      await Promise.all(
        batch.map(async (item) => {
          try {
            const res = await fetch(`/oembed?url=${encodeURIComponent(item.url)}&format=json`);
            if (res.ok) {
              const data = await res.json();
              if (data.title) {
                item.title = data.author_name ? `${data.title} (by ${data.author_name})` : data.title;
              }
            }
          } catch (e) {}
        })
      );
      progressInner.style.width = `${95 + Math.min(5, Math.round(((i + 5) / missingItems.length) * 5))}%`;
    }
  }

  progressInner.style.width = "100%";

  // 5. Present extracted items
  if (shortsList.length === 0) {
    statusEl.textContent = "No Shorts found in recent history. Make sure you are on YouTube Watch History.";
    return;
  }

  statusEl.textContent = `Found ${shortsList.length} Shorts in history`;

  previewList.style.display = "block";
  previewList.replaceChildren();

  shortsList.slice(0, 5).forEach((s, idx) => {
    const row = createEl("div", "margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:flex;align-items:center;gap:6px;");
    const numSpan = createEl("span", "color:#94a3b8;font-size:11px;font-variant-numeric:tabular-nums;min-width:18px;", `${idx + 1}.`);
    const labelSpan = createEl("span", "", s.title);
    row.appendChild(numSpan);
    row.appendChild(labelSpan);
    previewList.appendChild(row);
  });

  if (shortsList.length > 5) {
    const extra = createEl("div", "color:#64748b;font-size:11px;margin-top:6px;padding-top:4px;border-top:1px solid rgba(255,255,255,0.06);", `+ ${shortsList.length - 5} more videos`);
    previewList.appendChild(extra);
  }

  actionsBox.style.display = "flex";
  const jsonString = JSON.stringify(shortsList, null, 2);

  // Download Handler
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
    downloadBtn.textContent = "Downloaded!";
    setTimeout(() => {
      downloadBtn.textContent = "Download JSON";
    }, 2500);
  };

  // Copy Handler
  copyBtn.onclick = async () => {
    try {
      await navigator.clipboard.writeText(jsonString);
      copyBtn.textContent = "Copied to clipboard!";
    } catch (err) {
      const ta = document.createElement("textarea");
      ta.value = jsonString;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      copyBtn.textContent = "Copied to clipboard!";
    }
    setTimeout(() => {
      copyBtn.textContent = "Copy JSON to Clipboard";
    }, 2500);
  };
})();
