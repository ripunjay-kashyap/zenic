"use strict";

const $ = (id) => document.getElementById(id);
const form = $("chat-form");
const input = $("message-input");
const conversation = $("conversation");
const welcome = $("welcome");
const activity = $("activity");
const sendButton = $("send-button");
let busy = false;
let clockTimer;

const profileFields = [
  ["age", "Age"], ["weight_kg", "Weight"], ["height_cm", "Height"],
  ["gender", "Gender"], ["activity_level", "Activity"], ["goal", "Goal"],
  ["dietary_restrictions", "Diet"], ["experience_level", "Experience"],
  ["available_days", "Schedule"], ["equipment", "Equipment"],
];

function formatValue(key, value) {
  const text = String(value).replaceAll("_", " ");
  if (key === "weight_kg") return `${text} kg`;
  if (key === "height_cm") return `${text} cm`;
  if (key === "available_days") return `${text} days`;
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function setProfile(profile) {
  const required = ["age", "weight_kg", "height_cm", "gender", "activity_level", "goal"];
  const count = required.filter((key) => profile[key] !== undefined && profile[key] !== "").length;
  $("profile-count").textContent = `${count} of 6`;
  const percent = Math.round(count / required.length * 100);
  $("profile-fill").style.width = `${percent}%`;
  $("profile-progress").setAttribute("aria-valuenow", String(percent));
  $("profile-hint").hidden = Object.keys(profile).length > 0;
  const list = $("profile-fields");
  list.replaceChildren();
  for (const [key, label] of profileFields) {
    const value = profile[key];
    if (value === undefined || value === null || value === "") continue;
    const row = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = formatValue(key, value);
    row.append(dt, dd);
    list.append(row);
  }
}

function inlineMarkdown(parent, text) {
  const pieces = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|https:\/\/ods\.od\.nih\.gov\/factsheets\/[A-Za-z0-9-]+\/?)/g);
  for (const piece of pieces) {
    if (piece.startsWith("**") && piece.endsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = piece.slice(2, -2);
      parent.append(strong);
    } else if (piece.startsWith("`") && piece.endsWith("`")) {
      const code = document.createElement("code");
      code.textContent = piece.slice(1, -1);
      parent.append(code);
    } else if (piece.startsWith("*") && piece.endsWith("*")) {
      const emphasis = document.createElement("em");
      emphasis.textContent = piece.slice(1, -1);
      parent.append(emphasis);
    } else if (piece.startsWith("https://ods.od.nih.gov/factsheets/")) {
      const link = document.createElement("a");
      link.href = piece;
      link.textContent = piece;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      parent.append(link);
    } else {
      parent.append(document.createTextNode(piece));
    }
  }
}

function safeMarkdown(parent, text) {
  let list = null;
  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    const bullet = /^[-*] (.+)$/.exec(line);
    const numbered = /^\d+[.)] (.+)$/.exec(line);
    if (bullet || numbered) {
      const kind = bullet ? "ul" : "ol";
      if (!list || list.tagName.toLowerCase() !== kind) {
        list = document.createElement(kind);
        parent.append(list);
      }
      const li = document.createElement("li");
      inlineMarkdown(li, (bullet || numbered)[1]);
      list.append(li);
    } else {
      list = null;
      if (!line) continue;
      const isHeading = /^#{1,3} /.test(line);
      const element = document.createElement(isHeading ? "h3" : "p");
      inlineMarkdown(element, isHeading ? line.replace(/^#{1,3} /, "") : line);
      parent.append(element);
    }
  }
}

function addMessage(message, elapsedMs) {
  welcome.hidden = true;
  conversation.hidden = false;
  const row = document.createElement("article");
  row.className = `message ${message.role === "user" ? "user" : "assistant"}`;
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = message.role === "user" ? "◉" : "✳";
  const body = document.createElement("div");
  body.className = "message-body";
  const head = document.createElement("div");
  head.className = "message-head";
  head.textContent = message.role === "user" ? "You" : "Zenic";
  body.append(head);
  if (message.metrics && Object.keys(message.metrics).length) {
    const grid = document.createElement("div");
    grid.className = "metric-grid";
    for (const [key, label, unit] of [["tdee", "TDEE", "KCAL / DAY"], ["bmr", "BMR", "KCAL / DAY"], ["protein_g", "PROTEIN", "G / DAY AT TDEE"]]) {
      if (message.metrics[key] === undefined) continue;
      const card = document.createElement("div");
      card.className = "metric-card";
      const title = document.createElement("span");
      title.textContent = label;
      const value = document.createElement("strong");
      value.textContent = String(Math.round(message.metrics[key]));
      const sub = document.createElement("small");
      sub.textContent = unit;
      card.append(title, value, sub);
      grid.append(card);
    }
    body.append(grid);
  }
  const content = document.createElement("div");
  content.className = "message-content";
  safeMarkdown(content, message.content || "");
  body.append(content);
  if (elapsedMs && message.role !== "user") {
    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = `Completed in ${(elapsedMs / 1000).toFixed(1)}s`;
    body.append(meta);
  }
  row.append(avatar, body);
  conversation.append(row);
  row.scrollIntoView({behavior: "smooth", block: "end"});
}

function showError(text) {
  const card = document.createElement("div");
  card.className = "error-card";
  card.role = "alert";
  card.textContent = text;
  conversation.append(card);
  card.scrollIntoView({block: "end"});
}

function setBusy(value) {
  busy = value;
  input.disabled = value;
  sendButton.disabled = value;
  $("new-chat").disabled = value;
  activity.hidden = !value;
  if (value) {
    const started = performance.now();
    $("activity-clock").textContent = "0s";
    clockTimer = setInterval(() => {
      $("activity-clock").textContent = `${Math.floor((performance.now() - started) / 1000)}s`;
    }, 1000);
  } else {
    clearInterval(clockTimer);
    input.focus();
  }
}

async function sendMessage(prompt) {
  if (busy || !prompt.trim()) return;
  const message = prompt.trim();
  input.value = "";
  addMessage({role: "user", content: message});
  $("activity-label").textContent = "Checking your message";
  setBusy(true);
  try {
    const response = await fetch("/api/chat", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message}), credentials: "same-origin",
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.error || "The request could not be completed.");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
      let newline;
      while ((newline = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        if (!line) continue;
        const event = JSON.parse(line);
        if (event.type === "stage") $("activity-label").textContent = event.label;
        if (event.type === "error") throw new Error(event.message);
        if (event.type === "final") {
          addMessage(event.message, event.timings_ms?.total);
          setProfile(event.profile || {});
          $("plan-section").hidden = !event.download;
        }
      }
      if (done) break;
    }
  } catch (error) {
    showError(error.message || "Unable to connect. Please try again.");
  } finally {
    setBusy(false);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});
$("suggestions").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-prompt]");
  if (button) sendMessage(button.dataset.prompt);
});
$("new-chat").addEventListener("click", async () => {
  const response = await fetch("/api/reset", {method: "POST", credentials: "same-origin"});
  if (!response.ok) { showError("Could not start a new conversation."); return; }
  conversation.replaceChildren();
  conversation.hidden = true;
  welcome.hidden = false;
  $("plan-section").hidden = true;
  setProfile({});
  input.value = "";
  closeSidebar();
  input.focus();
});

function closeSidebar() {
  $("sidebar").classList.remove("open");
  $("sidebar-backdrop").hidden = true;
  $("open-sidebar").setAttribute("aria-expanded", "false");
}
$("open-sidebar").addEventListener("click", () => {
  $("sidebar").classList.add("open");
  $("sidebar-backdrop").hidden = false;
  $("open-sidebar").setAttribute("aria-expanded", "true");
});
$("close-sidebar").addEventListener("click", closeSidebar);
$("sidebar-backdrop").addEventListener("click", closeSidebar);
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeSidebar(); });

fetch("/api/session", {credentials: "same-origin"})
  .then((response) => response.json())
  .then((session) => {
    for (const message of session.messages || []) addMessage(message);
    setProfile(session.profile || {});
    $("plan-section").hidden = !session.download;
  })
  .catch(() => showError("Could not restore this conversation."));
