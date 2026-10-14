"use strict";

const byId = (id) => document.getElementById(id);
const cases = window.AGF_CASES;
const duration = 239.8125;
const pageSize = 4;
const methods = {
  agf: "AGF (ours)", opsdv: "OPSD-V", longlive: "LongLive",
  "self-forcing": "Self-Forcing", sgf: "SGF", "causal-forcing": "Causal Forcing",
};
const formatTime = (seconds) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let category = "All";
let page = 0;
let mode = "four";
let pairMethod = "sgf";
let players = [];
let activePlayer = null;
let observer;

for (const link of document.querySelectorAll("[data-resource]")) {
  const href = window.AGF_LINKS[link.dataset.resource];
  if (href) { link.href = href; link.removeAttribute("aria-disabled"); }
}

// Narration begins only after a user gesture; one playback context at a time.
const overview = byId("overview-video");
const overviewStart = byId("overview-start");
const overviewStatus = byId("overview-status");
overview.controls = false;
overviewStart.hidden = false;
overviewStart.addEventListener("click", async () => {
  overviewStatus.hidden = true;
  overview.controls = true;
  overviewStart.hidden = true;
  try {
    await overview.play();
  } catch {
    overviewStatus.hidden = false;
    overviewStart.hidden = false;
  }
});
overview.addEventListener("play", () => {
  activePlayer?.release();
  activePlayer = null;
  overviewStart.hidden = true;
  overviewStatus.hidden = true;
});
overview.addEventListener("error", () => {
  overviewStatus.hidden = false;
  overviewStart.hidden = false;
});
if ("IntersectionObserver" in window) {
  new IntersectionObserver(([entry]) => {
    if (!entry.isIntersecting) overview.pause();
  }).observe(overview);
}
document.addEventListener("visibilitychange", () => {
  if (document.hidden) overview.pause();
});

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function activate(player) {
  overview.pause();
  if (activePlayer && activePlayer !== player) activePlayer.release();
  activePlayer = player;
}

// Playback is local to one case. Every ready video can start immediately;
// slow or failed media never gates another method or another case.
class CasePlayer {
  constructor(card, item, selectedMethods) {
    this.card = card;
    this.item = item;
    this.methods = selectedMethods;
    this.time = 0;
    this.rate = 1;
    this.wanted = false;
    this.userPaused = false;
    this.solo = null;
    this.leader = null;
    this.frame = 0;
    this.lastUpdate = 0;
    this.visibility = 0;
    this.disposed = false;
    this.tiles = [];
    this.makeVideos();
    this.makeControls();
  }

  makeVideos() {
    const grid = element("div", `case-videos mode-${mode}`);
    for (const method of this.methods) {
      const tile = element("div", `video-cell${method === "agf" ? " ours" : ""}`);
      const video = document.createElement("video");
      video.muted = true;
      video.playsInline = true;
      video.preload = "none";
      video.poster = this.item.posters[method];
      video.width = 624;
      video.height = 360;
      video.setAttribute("aria-label", `${methods[method]}: ${this.item.title}`);
      const name = element("span", "method-label", methods[method]);
      const status = element("span", "video-status", "");
      status.setAttribute("aria-live", "polite");
      const solo = element("button", "solo-button", "▶");
      solo.type = "button";
      solo.title = `Play only ${methods[method]}`;
      solo.setAttribute("aria-label", `Play only ${methods[method]}: ${this.item.title}`);
      const state = { method, video, tile, status, joining: false, lastAlign: -Infinity };
      solo.addEventListener("click", () => this.play(method));
      video.addEventListener("loadedmetadata", () => {
        if (this.disposed) return;
        if (Math.abs(video.currentTime - this.time) > 0.08) video.currentTime = this.time;
      });
      video.addEventListener("canplay", () => {
        if (!this.wanted && !video.seeking) status.textContent = "";
        this.startReady(state);
      });
      video.addEventListener("seeked", () => {
        if (this.wanted) this.startReady(state);
        else if (video.readyState >= 2) status.textContent = "";
      });
      video.addEventListener("playing", () => {
        if (!this.wanted || !this.includes(state)) { video.pause(); return; }
        status.textContent = "";
        if (!this.leader) this.leader = video;
      });
      video.addEventListener("waiting", () => {
        if (this.wanted && this.includes(state)) status.textContent = "Buffering…";
      });
      video.addEventListener("error", () => {
        if (video.getAttribute("src")) status.textContent = "Could not load · press Play to retry";
      });
      tile.append(video, name, status, solo);
      grid.append(tile);
      this.tiles.push(state);
    }
    this.card.append(grid);
  }

  makeControls() {
    const controls = element("div", "case-controls");
    controls.setAttribute("role", "group");
    controls.setAttribute("aria-label", `Playback: ${this.item.title}`);
    this.playButton = element("button", "play-button", "Play case");
    this.playButton.type = "button";
    this.playButton.addEventListener("click", () => {
      if (this.solo) this.play();
      else if (this.wanted) this.pause(true);
      else this.play();
    });
    this.timeOutput = element("output", "case-time", "0:00 / 4:00");
    this.timeOutput.setAttribute("aria-live", "off");
    this.seekInput = document.createElement("input");
    Object.assign(this.seekInput, { type: "range", min: "0", max: String(duration - 0.0625), step: "0.0625", value: "0" });
    this.seekInput.setAttribute("aria-label", `Video time: ${this.item.title}`);
    this.seekInput.addEventListener("input", () => { this.timeOutput.textContent = `${formatTime(Number(this.seekInput.value))} / 4:00`; });
    this.seekInput.addEventListener("change", () => this.seek(Number(this.seekInput.value)));
    const speed = element("select", "case-speed");
    speed.setAttribute("aria-label", `Playback speed: ${this.item.title}`);
    for (const rate of [1, 2, 4, 8]) {
      const option = element("option", "", `${rate}×`);
      option.value = rate;
      option.selected = rate === this.rate;
      speed.append(option);
    }
    speed.addEventListener("change", () => {
      this.rate = Number(speed.value);
      this.tiles.forEach(({ video }) => { video.playbackRate = this.rate; });
    });
    const late = element("button", "late-button", "3:00");
    late.type = "button";
    late.setAttribute("aria-label", `Jump to three minutes: ${this.item.title}`);
    late.addEventListener("click", () => this.seek(180));
    controls.append(this.playButton, this.timeOutput, this.seekInput, speed, late);
    this.card.append(controls);
  }

  includes(state) { return this.solo === null || state.method === this.solo; }

  load(state) {
    const { video, status } = state;
    if (!video.getAttribute("src") || video.error) {
      status.textContent = "Loading…";
      video.preload = "auto";
      video.src = this.item.videos[state.method];
      video.load();
    }
    video.playbackRate = this.rate;
  }

  startReady(state) {
    const { video, status } = state;
    if (this.disposed || !this.wanted || !this.includes(state) || state.joining || video.readyState < 2) return;
    if (!video.paused) return;
    if (!video.seeking && Math.abs(video.currentTime - this.time) > 0.3) {
      video.currentTime = this.time;
      state.lastAlign = performance.now();
    }
    state.joining = true;
    video.play().catch((error) => {
      if (this.wanted && error.name !== "AbortError") {
        if (error.name === "NotAllowedError" && this.tiles.every((tile) => tile.video.paused)) this.pause(true);
        status.textContent = error.name === "NotAllowedError" ? "Press Play to start" : "Could not play · press Play to retry";
      }
    }).finally(() => {
      state.joining = false;
      if (!this.wanted || !this.includes(state) || this.disposed) video.pause();
    });
  }

  play(onlyMethod = null, automatic = false) {
    if (this.disposed || (automatic && this.userPaused)) return;
    activate(this);
    this.userPaused = false;
    this.solo = onlyMethod;
    this.wanted = true;
    this.leader = null;
    if (this.time >= duration - 0.1) this.time = 0;
    this.card.classList.add("is-playing");
    for (const state of this.tiles) {
      if (this.includes(state)) {
        this.load(state);
        this.startReady(state);
      } else {
        state.video.pause();
        state.status.textContent = "";
      }
    }
    this.update();
    cancelAnimationFrame(this.frame);
    this.tick();
  }

  tick() {
    if (!this.wanted || this.disposed) return;
    const ready = this.tiles.filter((s) => this.includes(s) && !s.video.paused && !s.video.seeking && s.video.readyState >= 2);
    if (!ready.some((s) => s.video === this.leader)) this.leader = ready[0]?.video || null;
    if (this.leader) this.time = this.leader.currentTime;
    const now = performance.now();
    if (now - this.lastUpdate > 250) {
      for (const state of ready) {
        const video = state.video;
        if (video !== this.leader && Math.abs(video.currentTime - this.time) > 0.45 && now - state.lastAlign > 2000) {
          video.currentTime = this.time;
          state.lastAlign = now;
        }
      }
      this.lastUpdate = now;
      this.update();
    }
    if (this.time >= duration - 0.1) { this.pause(); return; }
    this.frame = requestAnimationFrame(() => this.tick());
  }

  update() {
    this.playButton.textContent = this.solo ? `Play all ${this.tiles.length}` : this.wanted ? "Pause case" : "Play case";
    this.timeOutput.textContent = `${formatTime(this.time)} / 4:00`;
    if (document.activeElement !== this.seekInput) this.seekInput.value = this.time;
  }

  pause(manual = false) {
    if (this.leader && !this.leader.seeking) this.time = this.leader.currentTime;
    this.wanted = false;
    if (manual) this.userPaused = true;
    cancelAnimationFrame(this.frame);
    this.tiles.forEach(({ video, status }) => { video.pause(); status.textContent = ""; });
    this.card.classList.remove("is-playing");
    this.update();
  }

  seek(time) {
    activate(this);
    this.time = Math.max(0, Math.min(duration - 0.0625, time));
    this.leader = null;
    for (const state of this.tiles) {
      if (!this.includes(state)) continue;
      this.load(state);
      state.status.textContent = "Seeking…";
      state.lastAlign = performance.now();
      if (state.video.readyState >= 1) state.video.currentTime = this.time;
    }
    this.update();
  }

  release() {
    this.pause();
    for (const { video, status } of this.tiles) {
      video.removeAttribute("src");
      video.preload = "none";
      video.load();
      status.textContent = "";
    }
    this.leader = null;
  }

  destroy() { this.release(); this.disposed = true; }
}

function selectedMethods() {
  if (mode === "pair") return ["agf", pairMethod];
  const main = ["agf", "opsdv", "longlive", "self-forcing"];
  return mode === "six" ? [...main, "sgf", "causal-forcing"] : main;
}

function renderGallery() {
  observer?.disconnect();
  players.forEach((player) => player.destroy());
  players = [];
  activePlayer = null;
  const filtered = category === "All" ? cases : cases.filter((item) => item.category === category);
  const start = page * pageSize;
  const grid = byId("comparison-grid");
  grid.replaceChildren();
  for (const item of filtered.slice(start, start + pageSize)) {
    const card = element("article", "case-card");
    const heading = element("div", "case-heading");
    const title = element("h3", "", item.title);
    const tag = element("span", "case-category", item.category);
    heading.append(title, tag);
    card.append(heading);
    const player = new CasePlayer(card, item, selectedMethods());
    const details = element("details", "prompt");
    const summary = element("summary", "");
    const excerpt = item.prompt.match(/^.+?[.!?](?:\s|$)/)?.[0].trim() || item.prompt;
    summary.append(element("span", "prompt-label", "Prompt"), element("span", "prompt-excerpt", excerpt));
    details.append(summary, element("p", "", item.prompt));
    const download = element("a", "", "Download AGF video ↓");
    download.href = item.videos.agf;
    download.download = `agf-${item.id}.mp4`;
    details.append(download);
    card.append(details);
    grid.append(card);
    players.push(player);
  }
  byId("gallery-count").textContent = `${start + 1}–${Math.min(start + pageSize, filtered.length)} of ${filtered.length} cases`;
  for (const suffix of ["", "-top"]) {
    byId(`page-number${suffix}`).textContent = `${page + 1} / ${Math.ceil(filtered.length / pageSize)}`;
    byId(`page-prev${suffix}`).disabled = page === 0;
    byId(`page-next${suffix}`).disabled = start + pageSize >= filtered.length;
  }
  if ("IntersectionObserver" in window) {
    observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const player = players.find((p) => p.card === entry.target);
        if (player) player.visibility = entry.intersectionRatio;
      }
      if (activePlayer && activePlayer.visibility === 0) { activePlayer.release(); activePlayer = null; }
      if (document.hidden || reducedMotion || !overview.paused) return;
      const visible = players.filter((p) => p.visibility >= 0.4).sort((a, b) => b.visibility - a.visibility)[0];
      if (visible && visible !== activePlayer && !visible.userPaused) visible.play(null, true);
    }, { threshold: [0, 0.2, 0.4, 0.6, 0.8, 1] });
    players.forEach((player) => observer.observe(player.card));
  }
}

for (const name of ["All", "People", "Nature", "Imagination", "Animation"]) {
  const button = element("button", "", name);
  button.type = "button";
  button.setAttribute("aria-pressed", String(name === category));
  button.addEventListener("click", () => {
    category = name;
    page = 0;
    for (const child of byId("category-filters").children) child.setAttribute("aria-pressed", String(child === button));
    renderGallery();
  });
  byId("category-filters").append(button);
}
byId("comparison-mode").addEventListener("change", (event) => {
  mode = event.target.value;
  byId("pair-label").hidden = mode !== "pair";
  renderGallery();
});
byId("pair-method").addEventListener("change", (event) => { pairMethod = event.target.value; renderGallery(); });
for (const suffix of ["", "-top"]) {
  for (const [direction, delta] of [["prev", -1], ["next", 1]]) {
    byId(`page-${direction}${suffix}`).addEventListener("click", () => {
      page += delta;
      renderGallery();
      byId("gallery").scrollIntoView({ behavior: reducedMotion ? "instant" : "smooth" });
    });
  }
}
document.addEventListener("visibilitychange", () => { if (document.hidden) activePlayer?.pause(); });
byId("copy-citation").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(byId("bibtex").textContent);
    byId("citation-status").textContent = "BibTeX copied.";
  } catch {
    const range = document.createRange();
    range.selectNodeContents(byId("bibtex"));
    const selection = window.getSelection();
    selection.removeAllRanges(); selection.addRange(range);
    byId("citation-status").textContent = "Citation selected. Press Ctrl+C or ⌘C to copy.";
  }
});
renderGallery();
