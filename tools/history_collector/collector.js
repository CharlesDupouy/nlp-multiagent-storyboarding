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
    "position:fixed;top:24px;right:24px;width:400px;max-width:calc(100vw - 48px);background:#10131b;color:#f1f5f9;border:1px solid rgba(255,255,255,0.12);border-radius:12px;box-shadow:0 12px 36px rgba(0,0,0,0.75);z-index:9999999;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;padding:18px;box-sizing:border-box;line-height:1.45;"
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

  let isCancelled = false;
  let forceStop = false;

  const cleanup = () => {
    isCancelled = true;
    container.remove();
    window.removeEventListener("keydown", handleKeyDown);
  };
  closeBtn.onclick = cleanup;

  const handleKeyDown = (e) => {
    if (e.key === "Escape") cleanup();
  };
  window.addEventListener("keydown", handleKeyDown);

  header.appendChild(titleBox);
  header.appendChild(closeBtn);
  container.appendChild(header);

  // Status message
  const statusEl = createEl("div", "font-size:13px;color:#94a3b8;margin-bottom:10px;", "Scanning watch history...");
  container.appendChild(statusEl);

  // Progress Bar
  const progressBar = createEl("div", "height:4px;background:#1e293b;border-radius:2px;overflow:hidden;margin-bottom:12px;");
  const progressInner = createEl("div", "width:5%;height:100%;background:#e50914;transition:width 0.25s ease;");
  progressBar.appendChild(progressInner);
  container.appendChild(progressBar);

  // Live stop button (visible during scanning)
  const scanControlBox = createEl("div", "display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;");
  const liveCountBadge = createEl("span", "font-size:12px;color:#cbd5e1;font-variant-numeric:tabular-nums;", "0 Shorts found so far");
  const stopEarlyBtn = createEl("button", "background:#1e293b;color:#f1f5f9;border:1px solid rgba(255,255,255,0.08);padding:5px 12px;border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;transition:background 0.15s ease;", "Export Now");
  stopEarlyBtn.onmouseenter = () => { stopEarlyBtn.style.background = "#283548"; };
  stopEarlyBtn.onmouseleave = () => { stopEarlyBtn.style.background = "#1e293b"; };
  stopEarlyBtn.onclick = () => { forceStop = true; };
  scanControlBox.appendChild(liveCountBadge);
  scanControlBox.appendChild(stopEarlyBtn);
  container.appendChild(scanControlBox);

  // Preview List Container
  const previewList = createEl("div", "max-height:160px;overflow-y:auto;background:#090b10;border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:10px 12px;font-size:12px;color:#cbd5e1;margin-bottom:14px;display:none;");
  container.appendChild(previewList);

  // Actions Container
  const actionsBox = createEl("div", "display:none;flex-direction:column;gap:8px;");

  const downloadBtn = createEl("button", "background:#e50914;color:#fff;border:none;padding:10px 14px;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:6px;transition:opacity 0.15s ease;", "Download JSON");
  downloadBtn.onmouseenter = () => { downloadBtn.style.opacity = "0.9"; };
  downloadBtn.onmouseleave = () => { downloadBtn.style.opacity = "1"; };

  const copyBtn = createEl("button", "background:#1e293b;color:#f1f5f9;border:1px solid rgba(255,255,255,0.08);padding:9px 14px;border-radius:6px;font-size:13px;font-weight:500;cursor:pointer;transition:background 0.15s ease;", "Copy JSON to Clipboard");
  copyBtn.onmouseenter = () => { copyBtn.style.background = "#283548"; };
  copyBtn.onmouseleave = () => { copyBtn.style.background = "#1e293b"; };

  const noteEl = createEl("div", "font-size:11px;color:#64748b;text-align:center;margin-top:2px;", "HKUST COMP 4221 Study • Anonymous & Client-Side");

  actionsBox.appendChild(downloadBtn);
  actionsBox.appendChild(copyBtn);
  actionsBox.appendChild(noteEl);
  container.appendChild(actionsBox);

  document.body.appendChild(container);

  // Helpers to count distinct items in DOM
  function countDistinctShortsInDOM() {
    const anchors = document.querySelectorAll('a[href*="/shorts/"]');
    const ids = new Set();
    for (const a of anchors) {
      const href = a.getAttribute("href") || "";
      const m = href.match(/\/shorts\/([a-zA-Z0-9_-]+)/);
      if (m) ids.add(m[1]);
    }
    return ids.size;
  }

  function countTotalVideosInDOM() {
    return document.querySelectorAll("ytd-video-renderer, ytd-reel-item-renderer, ytd-rich-item-renderer, ytd-grid-video-renderer").length;
  }

  // Trigger YouTube infinite scroll loaders
  function triggerYouTubeScroll() {
    // 1. Scroll window and documentElement
    window.scrollTo(0, document.documentElement.scrollHeight);
    if (document.scrollingElement) {
      document.scrollingElement.scrollTop = document.scrollingElement.scrollHeight;
    }

    // 2. Bring YouTube's continuation sentinel element into view
    const continuations = document.querySelectorAll("ytd-continuation-item-renderer");
    if (continuations.length > 0) {
      const lastContinuation = continuations[continuations.length - 1];
      if (lastContinuation && lastContinuation.scrollIntoView) {
        lastContinuation.scrollIntoView({ block: "end", behavior: "instant" });
      }
    }

    // 3. Dispatch native scroll event to wake up YouTube listeners
    window.dispatchEvent(new Event("scroll"));
  }

  // 2. Deep Dynamic Scrolling with Network Grace Periods
  const TARGET_SHORTS = 100;
  const MAX_CYCLES = 35;
  let consecutiveStalls = 0;

  for (let cycle = 1; cycle <= MAX_CYCLES; cycle++) {
    if (isCancelled || forceStop) break;

    const currentCount = countDistinctShortsInDOM();
    const currentTotal = countTotalVideosInDOM();

    liveCountBadge.textContent = `${currentCount} Shorts found`;
    stopEarlyBtn.textContent = currentCount > 0 ? `Export Now (${currentCount})` : "Export Now";

    const progressPct = Math.min(85, Math.max(8, Math.round((currentCount / TARGET_SHORTS) * 85)));
    progressInner.style.width = `${progressPct}%`;

    // Stop if target reached
    if (currentCount >= TARGET_SHORTS) {
      break;
    }

    statusEl.textContent = `Scanning history... cycle ${cycle}/${MAX_CYCLES} (${currentCount} Shorts)`;
    triggerYouTubeScroll();

    // Active polling wait: wait up to 3.2 seconds for YouTube's network request to resolve and append DOM nodes
    const pollInterval = 300;
    let waitedMs = 0;
    let itemsAppended = false;

    while (waitedMs < 3200) {
      await new Promise((r) => setTimeout(r, pollInterval));
      waitedMs += pollInterval;

      if (isCancelled || forceStop) break;

      const newCount = countDistinctShortsInDOM();
      const newTotal = countTotalVideosInDOM();

      if (newCount > currentCount || newTotal > currentTotal) {
        itemsAppended = true;
        liveCountBadge.textContent = `${newCount} Shorts found`;
        stopEarlyBtn.textContent = `Export Now (${newCount})`;
        break;
      }

      // Show responsive waiting message
      if (waitedMs >= 1200) {
        statusEl.textContent = `Waiting for YouTube to load more... (${currentCount} Shorts)`;
        triggerYouTubeScroll();
      }
    }

    if (itemsAppended) {
      consecutiveStalls = 0;
    } else {
      consecutiveStalls++;
      // Check if YouTube still has a continuation spinner/renderer waiting
      const hasContinuation = document.querySelector("ytd-continuation-item-renderer") !== null;

      // Only terminate if no new items loaded after multiple consecutive attempts AND no continuation exists
      const maxAllowedStalls = hasContinuation ? 5 : 3;
      if (consecutiveStalls >= maxAllowedStalls) {
        // End of user history reached
        break;
      }
    }
  }

  if (isCancelled) return;

  scanControlBox.style.display = "none";
  statusEl.textContent = "Extracting Shorts details...";
  progressInner.style.width = "90%";

  // 3. Extract shorts details from page
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

  // 4. Enrich missing titles via public YouTube oEmbed in batches
  const missingItems = shortsList.filter((s) => s.title.startsWith("YouTube Short (") || !s.title);
  if (missingItems.length > 0) {
    statusEl.textContent = `Resolving titles (${missingItems.length} videos)...`;
    for (let i = 0; i < missingItems.length; i += 6) {
      if (isCancelled) return;
      const batch = missingItems.slice(i, i + 6);
      await Promise.all(
        batch.map(async (item) => {
          try {
            const controller = new AbortController();
            const tid = setTimeout(() => controller.abort(), 3500);
            const res = await fetch(`/oembed?url=${encodeURIComponent(item.url)}&format=json`, { signal: controller.signal });
            clearTimeout(tid);
            if (res.ok) {
              const data = await res.json();
              if (data.title) {
                item.title = data.author_name ? `${data.title} (by ${data.author_name})` : data.title;
              }
            }
          } catch (e) {}
        })
      );
      progressInner.style.width = `${90 + Math.min(10, Math.round(((i + 6) / missingItems.length) * 10))}%`;
    }
  }

  progressInner.style.width = "100%";

  // 5. Present extracted items
  if (shortsList.length === 0) {
    statusEl.textContent = "No Shorts found in history. Make sure you are on YouTube Watch History.";
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
  downloadBtn.textContent = `Download JSON (${shortsList.length} Shorts)`;

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
      downloadBtn.textContent = `Download JSON (${shortsList.length} Shorts)`;
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
