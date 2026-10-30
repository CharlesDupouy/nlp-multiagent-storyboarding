/**
 * YouTube Shorts Watch History Exporter (COMP 4221 NLP Study)
 *
 * This script runs locally in the user's browser while viewing https://www.youtube.com/feed/history.
 * It automatically scrolls down to load recent entries, identifies all YouTube Shorts,
 * extracts their titles and canonical URLs in chronological order (most recent first),
 * and provides 1-click Download and Copy buttons.
 *
 * 100% Client-side. No passwords, tokens, or personal identifiers are collected.
 */
(async function runShortsCollector() {
  // 1. Check if user is on YouTube History page
  if (!window.location.href.includes("youtube.com")) {
    alert("⚠️ Please open YouTube first!\n\nNavigate to https://www.youtube.com/feed/history and click this bookmark again.");
    window.location.href = "https://www.youtube.com/feed/history";
    return;
  }

  // Remove existing modal if already open
  const existingModal = document.getElementById("yt-shorts-collector-modal");
  if (existingModal) existingModal.remove();

  // 2. Inject sleek UI overlay into page
  const container = document.createElement("div");
  container.id = "yt-shorts-collector-modal";
  container.style.cssText = `
    position: fixed;
    top: 24px;
    right: 24px;
    width: 380px;
    max-width: calc(100vw - 48px);
    background: #0f0f13;
    color: #f1f1f1;
    border: 1px solid #ff0033;
    border-radius: 14px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.85);
    z-index: 9999999;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    padding: 20px;
    box-sizing: border-box;
    line-height: 1.4;
  `;

  container.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
      <div style="font-weight: 700; font-size: 16px; color: #fff; display: flex; align-items: center; gap: 8px;">
        <span style="background: #ff0033; color: #fff; font-size: 11px; font-weight: 800; padding: 2px 6px; border-radius: 4px;">SHORTS</span>
        <span>History Exporter</span>
      </div>
      <button id="yt-collector-close-btn" style="background: none; border: none; color: #888; font-size: 18px; cursor: pointer; padding: 0 4px;">✕</button>
    </div>
    <div id="yt-collector-status" style="font-size: 13px; color: #aaa; margin-bottom: 14px;">
      Scanning your watch history...
    </div>
    <div id="yt-collector-progress-bar" style="height: 4px; background: #222; border-radius: 2px; overflow: hidden; margin-bottom: 14px;">
      <div id="yt-collector-progress-inner" style="width: 15%; height: 100%; background: #ff0033; transition: width 0.3s ease;"></div>
    </div>
    <div id="yt-collector-preview-list" style="max-height: 140px; overflow-y: auto; background: #16161c; border-radius: 8px; padding: 10px; font-size: 12px; color: #ccc; margin-bottom: 14px; display: none;">
    </div>
    <div id="yt-collector-actions" style="display: none; flex-direction: column; gap: 8px;">
      <button id="yt-collector-download-btn" style="background: #ff0033; color: #fff; border: none; padding: 10px 14px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px;">
        📥 Download my_shorts_history.json
      </button>
      <button id="yt-collector-copy-btn" style="background: #272733; color: #eee; border: 1px solid #3c3c4f; padding: 9px 14px; border-radius: 8px; font-size: 13px; font-weight: 500; cursor: pointer;">
        📋 Copy JSON to Clipboard
      </button>
      <div style="font-size: 11px; color: #777; text-align: center; margin-top: 4px;">
        COMP 4221 NLP Study • Anonymous & Private
      </div>
    </div>
  `;

  document.body.appendChild(container);

  const statusEl = document.getElementById("yt-collector-status");
  const progressInner = document.getElementById("yt-collector-progress-inner");
  const previewList = document.getElementById("yt-collector-preview-list");
  const actionsEl = document.getElementById("yt-collector-actions");
  const closeBtn = document.getElementById("yt-collector-close-btn");
  closeBtn.onclick = () => container.remove();

  // 3. Smooth auto-scroll to fetch recent history
  const SCROLL_STEPS = 5;
  for (let i = 1; i <= SCROLL_STEPS; i++) {
    statusEl.innerHTML = `Loading history... scrolling (${i}/${SCROLL_STEPS})`;
    progressInner.style.width = `${(i / (SCROLL_STEPS + 1)) * 100}%`;
    window.scrollBy({ top: 1800, behavior: "smooth" });
    await new Promise((r) => setTimeout(r, 650));
  }

  statusEl.innerHTML = "Extracting YouTube Shorts...";
  progressInner.style.width = "95%";

  // 4. Extract shorts items from page
  const shortsList = [];
  const seenUrls = new Set();

  // Strategy A: Find all anchor tags pointing to /shorts/
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

    // Remove YouTube duration / metadata chatter if present in aria-label
    title = title.replace(/\s+by\s+.*$/i, "").replace(/\s+-\s+play\s+short$/i, "").trim();

    seenUrls.add(canonicalUrl);
    shortsList.push({ title, url: canonicalUrl });
  }

  progressInner.style.width = "100%";

  // 5. Present results in modal
  if (shortsList.length === 0) {
    statusEl.innerHTML = `
      <span style="color: #ff9900;">⚠️ No Shorts found yet.</span><br>
      <span style="font-size: 11px; color: #888;">Make sure you are on <a href="/feed/history" style="color: #ff0033;">Watch History</a> and have watched YouTube Shorts recently while logged in.</span>
    `;
    return;
  }

  statusEl.innerHTML = `✅ Successfully found <strong>${shortsList.length} recent Shorts</strong>!`;

  // Render top preview items
  previewList.style.display = "block";
  previewList.innerHTML = shortsList
    .slice(0, 5)
    .map(
      (s, idx) =>
        `<div style="margin-bottom: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
          <span style="color: #ff0033; font-weight: bold;">#${idx + 1}</span> ${s.title}
        </div>`
    )
    .join("") + (shortsList.length > 5 ? `<div style="color: #666; font-size: 11px;">+ ${shortsList.length - 5} more items</div>` : "");

  actionsEl.style.display = "flex";

  const jsonString = JSON.stringify(shortsList, null, 2);

  // Download Button
  const downloadBtn = document.getElementById("yt-collector-download-btn");
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
    downloadBtn.innerHTML = "✅ Downloaded!";
    setTimeout(() => {
      downloadBtn.innerHTML = "📥 Download my_shorts_history.json";
    }, 2500);
  };

  // Copy to Clipboard Button
  const copyBtn = document.getElementById("yt-collector-copy-btn");
  copyBtn.onclick = async () => {
    try {
      await navigator.clipboard.writeText(jsonString);
      copyBtn.innerHTML = "✅ Copied to clipboard!";
    } catch (err) {
      // Fallback
      const ta = document.createElement("textarea");
      ta.value = jsonString;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      copyBtn.innerHTML = "✅ Copied to clipboard!";
    }
    setTimeout(() => {
      copyBtn.innerHTML = "📋 Copy JSON to Clipboard";
    }, 2500);
  };
})();
