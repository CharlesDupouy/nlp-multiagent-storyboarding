# Participant Guide — Step 1: Export Your YouTube Shorts Watch History

Thank you for participating in our **COMP 4221 (HKUST)** research study on personalized short-video storyboarding!  
Before we generate your 6 personalized prototype videos, we need a quick snapshot of the YouTube Shorts you recently watched (~2 minutes).

---

## Privacy & Data Safety
- The exporter runs **100% locally inside your browser** (client-side only).
- It **never** accesses your Google password, emails, account settings, or personal details.
- It only collects the **titles and video links** of recently watched YouTube Shorts shown on your history page.

---

## How to Export Your History (3 Steps)

> **Important:** Please do this on a **computer / desktop browser** (Chrome, Edge, Firefox, Brave, or Safari). The YouTube mobile app does not support bookmarks/console scripts.

### 1. Open our collector website on GitHub
Go to:  
👉 **https://charlesdupouy.github.io/nlp-multiagent-storyboarding/**

### 2. Follow the instructions shown on the website
- First, open **YouTube History** (`youtube.com/feed/history`) in the same browser while signed in to your usual YouTube account.
- Choose **either** method on the GitHub website:
  - **Bookmarks Bar (Recommended):** Drag the **"Export Shorts History"** button to your browser's bookmarks bar (`Ctrl+Shift+B` or `Cmd+Shift+B` to show it), switch to your `youtube.com/feed/history` tab, and click the bookmark.
  - **Developer Console (Alternative):** Copy the 1-line script from the website, open `youtube.com/feed/history`, press `F12` (or right-click → **Inspect** → **Console**), paste the script, and press `Enter`.
- A small floating overlay will scan your history and collect up to **90 recent YouTube Shorts** automatically (or click **"Export Now"** once ready).
- Your browser will automatically download a file named **`youtube_shorts_history.json`**.
- *(Optional)* You can drop `youtube_shorts_history.json` into the **"Verify Exported JSON"** box at the bottom of the website to check it looks valid.

### 3. Send the downloaded JSON back to us
Send **`youtube_shorts_history.json`** directly to **the person who asked you to participate** (Charles, Antoine, or Max).

---

Once we receive your file, we will prepare your **6 short videos** and send you a short follow-up rating link (**~15–20 minutes total**). Thank you!
