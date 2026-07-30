const state = {
  authMode: "login",
  user: null,
  conversationId: null,
  sending: false,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function cookie(name) {
  return document.cookie
    .split(";")
    .map((part) => part.trim())
    .find((part) => part.startsWith(`${name}=`))
    ?.split("=")
    .slice(1)
    .join("=") || "";
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  const csrf = cookie("anlu_csrf");
  if (csrf && !["GET", "HEAD"].includes((options.method || "GET").toUpperCase())) {
    headers["X-CSRF-Token"] = decodeURIComponent(csrf);
  }
  const response = await fetch(path, { credentials: "same-origin", ...options, headers });
  if (response.status === 204) return null;
  const body = await response.json().catch(() => ({ detail: "Unexpected server response" }));
  if (!response.ok) throw new Error(errorMessage(body.detail));
  return body;
}

function errorMessage(detail) {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => (typeof item === "string" ? item : item?.msg))
      .filter((message) => typeof message === "string" && message.trim())
      .map((message) => message.replace(/^Value error,\s*/i, ""));
    if (messages.length) return [...new Set(messages)].join(" ");
  }
  if (detail && typeof detail === "object" && typeof detail.message === "string") {
    return detail.message;
  }
  return "Request failed";
}

function validateCredentials(email, password, registering) {
  const validEmail = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  if (!validEmail) return "Enter a valid email address.";
  if (!password) return "Enter your password.";
  if (!registering) return "";
  if (password.length < 12) return "Use at least 12 characters for your password.";
  const characterClasses = [/[a-z]/.test(password), /[A-Z]/.test(password), /\d/.test(password)];
  if (characterClasses.filter(Boolean).length < 2) {
    return "Use at least two of: uppercase letters, lowercase letters, and numbers.";
  }
  return "";
}

function setAuthMode(mode) {
  state.authMode = mode;
  const register = mode === "register";
  $("#login-tab").classList.toggle("active", !register);
  $("#register-tab").classList.toggle("active", register);
  $("#login-tab").setAttribute("aria-selected", String(!register));
  $("#register-tab").setAttribute("aria-selected", String(register));
  $("#auth-title").textContent = register ? "Create your account" : "Welcome back";
  $("#auth-subtitle").textContent = register
    ? "Your questions stay private by default."
    : "Sign in to start an evidence-grounded conversation.";
  $("#password").autocomplete = register ? "new-password" : "current-password";
  $("#password-hint").classList.toggle("hidden", !register);
  $("#terms-row").classList.toggle("hidden", !register);
  $("#auth-error").textContent = "";
}

function openAuthModal(mode = "login") {
  setAuthMode(mode);
  $("#auth-modal").classList.remove("hidden");
  document.body.classList.add("modal-open");
  window.setTimeout(() => $("#email").focus(), 0);
}

function closeAuthModal() {
  $("#auth-modal").classList.add("hidden");
  document.body.classList.remove("modal-open");
  $("#auth-error").textContent = "";
}

function showApp(user) {
  state.user = user;
  closeAuthModal();
  $("#medical-disclaimer").checked = false;
  $("#chat-consent-error").textContent = "";
  $("#auth-shell").classList.add("hidden");
  $("#app-shell").classList.remove("hidden");
  $("#user-email").textContent = user.email;
  $("#avatar").textContent = user.email.charAt(0).toUpperCase();
  $(".privacy-chip").innerHTML = user.history_enabled
    ? "<span></span> Encrypted history on"
    : "<span></span> History off";
  $("#history-status").textContent = user.history_enabled
    ? "This deployment stores app-level encrypted conversation history."
    : "Not stored by this deployment.";
}

function showAuth() {
  state.user = null;
  closeAuthModal();
  $("#app-shell").classList.add("hidden");
  $("#auth-shell").classList.remove("hidden");
}

async function loadSession() {
  try {
    showApp(await api("/api/me"));
  } catch (_) {
    showAuth();
  }
}

async function loadRuntimeStatus() {
  try {
    const runtime = await api("/health/ready");
    $("#runtime-pill").classList.remove("hidden");
    if (runtime.runtime_mode === "local_experimental") {
      $("#runtime-status-label").textContent = "Local experimental mode";
      $("#processing-disclosure").textContent =
        "This local build uses deterministic mock inference and embeddings. Questions stay on this laptop, conversation history is disabled, and health text is excluded from application logs. Hosted deployments use a separately configured inference provider.";
      $("#terms-disclosure").textContent =
        "I understand this provides educational possibilities and examples, not diagnosis or personalized treatment, and that this local experiment uses deterministic mock inference rather than a clinical model.";
      return;
    }
    $("#runtime-status-label").textContent = "Private V32 model active";
    const retrievalDisclosure =
      runtime.embedding_runtime === "deterministic_mock"
        ? "Approved-source retrieval currently uses deterministic local lexical and feature-hash ranking, not a neural embedding service."
        : `Approved-source retrieval uses the configured ${runtime.embedding_runtime} embedding provider.`;
    $("#processing-disclosure").textContent =
      `Questions are processed by the configured private model endpoint. ${retrievalDisclosure} Conversation history is disabled unless explicitly enabled, and health text is excluded from application logs.`;
    $("#terms-disclosure").textContent =
      "I understand this provides educational possibilities and examples, not diagnosis or personalized treatment, and that my question is processed by the configured private model endpoint.";
  } catch (_) {
    // The main application will surface readiness failures when an action is attempted.
  }
}

function resetConversation() {
  state.conversationId = null;
  $("#messages").replaceChildren();
  $("#welcome").classList.remove("hidden");
  $("#urgent-note").classList.add("hidden");
  $("#message-input").value = "";
  $("#message-input").focus();
}

function addUserMessage(text) {
  const wrapper = document.createElement("div");
  wrapper.className = "message user";
  const bubble = document.createElement("div");
  bubble.className = "user-bubble";
  bubble.textContent = text;
  wrapper.appendChild(bubble);
  $("#messages").appendChild(wrapper);
}

function addTyping() {
  const wrapper = document.createElement("div");
  wrapper.id = "typing-message";
  wrapper.className = "message assistant-message";
  const mark = document.createElement("div");
  mark.className = "assistant-mark";
  mark.textContent = "+";
  const typing = document.createElement("div");
  typing.className = "typing";
  typing.innerHTML = "<i></i><i></i><i></i>";
  wrapper.append(mark, typing);
  $("#messages").appendChild(wrapper);
}

const answerHeadings = new Set([
  "What to do now",
  "What this may mean",
  "Possible explanations",
  "Concrete examples",
  "What to watch",
  "Traditional Chinese medicine perspective",
  "Helpful follow-up questions",
  "Evidence note",
]);

function renderAnswerText(container, text) {
  const sections = text.split("\n");
  let paragraph = [];
  const flush = () => {
    if (!paragraph.length) return;
    const p = document.createElement("p");
    p.textContent = paragraph.join("\n").trim();
    if (p.textContent) container.appendChild(p);
    paragraph = [];
  };
  sections.forEach((line) => {
    const trimmed = line.trim();
    if (answerHeadings.has(trimmed)) {
      flush();
      const heading = document.createElement("h4");
      heading.textContent = trimmed;
      container.appendChild(heading);
    } else if (!trimmed) {
      flush();
    } else {
      paragraph.push(line);
    }
  });
  flush();
}

function addAssistantMessage(payload) {
  $("#typing-message")?.remove();
  $("#urgent-note").textContent = "";
  $("#urgent-note").classList.add("hidden");
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant-message";
  const mark = document.createElement("div");
  mark.className = "assistant-mark";
  mark.textContent = "+";
  const content = document.createElement("div");

  const head = document.createElement("div");
  head.className = "answer-head";
  const name = document.createElement("b");
  name.textContent = "Anlu Health";
  const urgency = document.createElement("span");
  urgency.className = `urgency ${payload.urgency}`;
  urgency.textContent = payload.urgency;
  head.append(name, urgency);

  const body = document.createElement("div");
  body.className = "answer-body";
  renderAnswerText(body, payload.answer);
  const evidence = document.createElement("div");
  const evidenceStatus = payload.evidence_status || "model_rejected";
  evidence.className = `evidence-state ${evidenceStatus}`;
  const evidenceLabel = document.createElement("b");
  evidenceLabel.textContent = "Evidence check";
  const evidenceNotice = document.createElement("span");
  evidenceNotice.textContent =
    payload.evidence_notice || "The evidence state for this response is unavailable.";
  evidence.append(evidenceLabel, evidenceNotice);
  content.append(head, evidence, body);

  if (payload.sources?.length) {
    const sources = document.createElement("div");
    sources.className = "sources";
    const label = document.createElement("b");
    label.textContent = "Sources used in this answer";
    sources.appendChild(label);
    payload.sources.forEach((source) => {
      const link = document.createElement("a");
      link.className = "source-card";
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      const details = document.createElement("span");
      const title = document.createElement("strong");
      title.textContent = `${source.id} · ${source.title}`;
      const publisher = document.createElement("small");
      publisher.textContent = `${source.publisher} · reviewed ${source.reviewed_on}`;
      details.append(title, publisher);
      const tier = document.createElement("span");
      tier.className = "source-tier";
      tier.textContent = source.evidence_tier.replaceAll("_", " ");
      link.append(details, tier);
      sources.appendChild(link);
      if (source.license?.includes("Open Government Licence")) {
        const attribution = document.createElement("small");
        attribution.className = "source-attribution";
        attribution.append("Information from the NHS website is licensed under the ");
        const licenceLink = document.createElement("a");
        licenceLink.href =
          "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/";
        licenceLink.target = "_blank";
        licenceLink.rel = "noopener noreferrer";
        licenceLink.textContent = "Open Government Licence v3.0";
        attribution.append(licenceLink, ".");
        sources.appendChild(attribution);
      }
    });
    content.appendChild(sources);
  }

  const disclaimer = document.createElement("p");
  disclaimer.className = "answer-disclaimer";
  disclaimer.textContent = payload.disclaimer;
  content.appendChild(disclaimer);
  wrapper.append(mark, content);
  $("#messages").appendChild(wrapper);

  if (["emergency", "urgent"].includes(payload.urgency)) {
    $("#urgent-note").textContent =
      payload.urgency === "emergency"
        ? "Please act on the emergency guidance above now."
        : "This answer recommends prompt in-person assessment; do not rely on chat alone.";
    $("#urgent-note").classList.remove("hidden");
  }
}

async function sendMessage(text) {
  if (state.sending || !text.trim()) return;
  if (!$("#medical-disclaimer").checked) {
    $("#chat-consent-error").textContent =
      "Please agree to the educational-use acknowledgement before using the chatbot.";
    $("#medical-disclaimer").focus();
    return;
  }
  $("#chat-consent-error").textContent = "";
  state.sending = true;
  $("#send-button").disabled = true;
  $("#welcome").classList.add("hidden");
  addUserMessage(text.trim());
  addTyping();
  $("#message-input").value = "";
  resizeComposer();
  $("#chat-scroll").scrollTop = $("#chat-scroll").scrollHeight;
  try {
    const payload = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        message: text.trim(),
        care_mode: $("#care-mode").value,
        conversation_id: state.conversationId,
        medical_disclaimer_accepted: true,
      }),
    });
    state.conversationId = payload.conversation_id || state.conversationId;
    addAssistantMessage(payload);
  } catch (error) {
    $("#typing-message")?.remove();
    addAssistantMessage({
      answer: `${error.message}. No medical answer was generated. Please try again or contact a qualified clinician.`,
      urgency: "routine",
      evidence_status: "model_rejected",
      evidence_notice: "No verified response was produced.",
      sources: [],
      disclaimer: "The service could not produce a verified response.",
    });
  } finally {
    state.sending = false;
    $("#send-button").disabled = false;
    $("#chat-scroll").scrollTop = $("#chat-scroll").scrollHeight;
  }
}

function resizeComposer() {
  const input = $("#message-input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 130)}px`;
  input.style.overflowY = input.scrollHeight > 130 ? "auto" : "hidden";
}

$("#login-tab").addEventListener("click", () => setAuthMode("login"));
$("#register-tab").addEventListener("click", () => setAuthMode("register"));
$$("[data-auth-mode]").forEach((button) => {
  button.addEventListener("click", () => openAuthModal(button.dataset.authMode));
});
$("#close-auth").addEventListener("click", closeAuthModal);
$("#auth-backdrop").addEventListener("click", closeAuthModal);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !$("#auth-modal").classList.contains("hidden")) closeAuthModal();
});

$("#auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const email = $("#email").value.trim();
  const password = $("#password").value;
  const submit = $("#auth-submit");
  $("#auth-error").textContent = "";
  const validationError = validateCredentials(email, password, state.authMode === "register");
  if (validationError) {
    $("#auth-error").textContent = validationError;
    return;
  }
  if (state.authMode === "register" && !$("#terms").checked) {
    $("#auth-error").textContent = "Please accept the informational-use terms.";
    return;
  }
  submit.disabled = true;
  try {
    const path = state.authMode === "register" ? "/api/auth/register" : "/api/auth/login";
    const payload = { email, password };
    if (state.authMode === "register") payload.terms_accepted = true;
    showApp(await api(path, { method: "POST", body: JSON.stringify(payload) }));
  } catch (error) {
    $("#auth-error").textContent = error.message;
  } finally {
    submit.disabled = false;
  }
});

$("#chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage($("#message-input").value);
});

$("#message-input").addEventListener("input", resizeComposer);
$("#medical-disclaimer").addEventListener("change", () => {
  if ($("#medical-disclaimer").checked) $("#chat-consent-error").textContent = "";
});
$("#message-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#chat-form").requestSubmit();
  }
});

$$('.prompt-card').forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

$("#new-chat").addEventListener("click", resetConversation);
$("#menu-button").addEventListener("click", () => $(".sidebar").classList.toggle("open"));

$("#logout-button").addEventListener("click", async () => {
  try { await api("/api/auth/logout", { method: "POST", body: "{}" }); } catch (_) { /* clear UI */ }
  resetConversation();
  showAuth();
});

$("#settings-button").addEventListener("click", () => $("#settings-dialog").showModal());
$("#close-settings").addEventListener("click", () => $("#settings-dialog").close());

$("#delete-history").addEventListener("click", async () => {
  if (!confirm("Delete all stored conversation history?")) return;
  try {
    await api("/api/history", { method: "DELETE", body: "{}" });
    $("#settings-message").textContent = "Stored conversation history was deleted.";
    resetConversation();
  } catch (error) {
    $("#settings-message").textContent = error.message;
  }
});

$("#delete-account").addEventListener("click", async () => {
  const password = prompt("Enter your password to permanently delete this account:");
  if (!password) return;
  try {
    await api("/api/account", { method: "DELETE", body: JSON.stringify({ password }) });
    $("#settings-dialog").close();
    resetConversation();
    showAuth();
  } catch (error) {
    $("#settings-message").textContent = error.message;
  }
});

setAuthMode("login");
loadRuntimeStatus();
loadSession();
