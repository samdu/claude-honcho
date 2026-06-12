import { Honcho } from "@honcho-ai/sdk";
import { loadConfig, getSessionName, getHonchoClientOptions, isPluginEnabled, getCachedStdin, getObservationMode, isLoggingEnabled } from "../config.js";
import {
  getCachedUserContext,
  getStaleCachedUserContext,
  isContextCacheStale,
  setCachedUserContext,
  getMessageCount,
  incrementMessageCount,
  shouldRefreshKnowledgeGraph,
  markKnowledgeGraphRefreshed,
  getInstanceIdForCwd,
  queueMessage,
  getInjectedHashesForSession,
  recordInjectedHashes,
} from "../cache.js";
import { logHook, logApiCall, logCache, setLogContext } from "../log.js";
import { visContextLine, visSkipMessage, addSystemMessage, verboseApiResult, verboseList } from "../visual.js";
import { normalizeLine, hashLine } from "../dedup.js";
import { honchoSessionUrl } from "../styles.js";

interface HookInput {
  prompt?: string;
  cwd?: string;
  session_id?: string;
  workspace_roots?: string[];
}

// Patterns to skip context injection
const SKIP_CONTEXT_PATTERNS = [
  /^(yes|no|ok|sure|thanks|y|n|yep|nope|yeah|nah|continue|go ahead|do it|proceed)$/i,
  /^\//, // slash commands
];

const FETCH_TIMEOUT_MS = 4000;

/**
 * Extract meaningful topics from a prompt for semantic search.
 * Returns terms that are high-signal for conclusion matching.
 */
export function extractTopics(prompt: string): string[] {
  const topics: string[] = [];

  // File paths (high signal)
  const filePaths = prompt.match(/[\w\-\/\.]+\.(ts|tsx|js|jsx|py|rs|go|md|json|yaml|yml|toml|sql)/gi) || [];
  topics.push(...filePaths.slice(0, 5));

  // Quoted strings (explicit references)
  const quoted = prompt.match(/"([^"]+)"/g)?.map(q => q.slice(1, -1)) || [];
  topics.push(...quoted.slice(0, 3));

  // Technical terms
  const techTerms = prompt.match(/\b(react|vue|svelte|angular|elysia|express|fastapi|django|flask|postgres|redis|docker|kubernetes|bun|node|deno|typescript|python|rust|go|graphql|rest|api|auth|oauth|jwt|stripe|webhook|honcho|mcp|claude|cursor|sentry)\b/gi) || [];
  topics.push(...[...new Set(techTerms.map(t => t.toLowerCase()))].slice(0, 5));

  // Error patterns
  const errors = prompt.match(/error[:\s]+[\w\s]+|failed[:\s]+[\w\s]+|exception[:\s]+[\w\s]+/gi) || [];
  topics.push(...errors.slice(0, 2));

  // No keyword fallback: extracted word lists are English-only and produce
  // low-signal queries for other languages. Callers fall back to the raw
  // prompt, which embeds better for semantic search anyway.
  return [...new Set(topics)];
}

function shouldSkipContextRetrieval(prompt: string): boolean {
  return SKIP_CONTEXT_PATTERNS.some((p) => p.test(prompt.trim()));
}

function formatSessionLink(sessionUrl: string): string {
  return `view your session in honcho GUI: ${sessionUrl}`;
}

/**
 * UserPromptSubmit hook — serves cached context instantly, refreshes when stale.
 *
 * Context lifecycle:
 *   SessionStart  -> warms cache (parallel API calls, 30s budget)
 *   UserPrompt    -> serves cache; refreshes (with 4s timeout) when TTL expires or message threshold hit
 *   PreCompact    -> re-warms cache before context window reset
 *
 * On refresh failure, silently falls back to stale cache.
 * On no cache at all, exits silently — context will arrive next turn.
 */
export async function handleUserPrompt(): Promise<void> {
  const config = loadConfig();
  if (!config) {
    process.exit(0);
  }

  if (!isPluginEnabled()) {
    process.exit(0);
  }

  let hookInput: HookInput = {};
  try {
    const input = getCachedStdin() ?? await Bun.stdin.text();
    if (input.trim()) {
      hookInput = JSON.parse(input);
    }
  } catch {
    process.exit(0);
  }

  const prompt = hookInput.prompt || "";
  const cwd = hookInput.workspace_roots?.[0] || hookInput.cwd || process.cwd();
  const instanceId = hookInput.session_id || getInstanceIdForCwd(cwd);
  const dedupSessionId = hookInput.session_id;
  const sessionName = getSessionName(cwd, instanceId || undefined);

  setLogContext(cwd, sessionName);

  if (!prompt.trim()) {
    process.exit(0);
  }

  logHook("user-prompt", `Prompt received (${prompt.length} chars)`);

  // Queue user prompt for upload at session-end (instant, no network)
  if (config.saveMessages !== false) {
    queueMessage(prompt, config.peerName, cwd, instanceId || undefined);
  }

  // Track message count for threshold-based refresh
  const messageCountBefore = getMessageCount();
  incrementMessageCount();
  const shouldShowSessionLink = messageCountBefore === 0;

  // Build session link lazily — only materialized on first message
  const sessionLink = shouldShowSessionLink
    ? formatSessionLink(honchoSessionUrl(config.workspace, sessionName))
    : undefined;

  // Skip trivial prompts — no context needed for "y", "ok", etc.
  if (shouldSkipContextRetrieval(prompt)) {
    logHook("user-prompt", "Skipping context (trivial prompt)");
    visSkipMessage("user-prompt", sessionLink ? `${sessionLink} · trivial prompt` : "trivial prompt");
    process.exit(0);
  }

  // Decide whether to refresh: TTL expired or message threshold hit
  const forceRefresh = shouldRefreshKnowledgeGraph();
  const cachedContext = getCachedUserContext();
  const cacheIsStale = isContextCacheStale();

  if (cachedContext && !cacheIsStale && !forceRefresh) {
    // Fresh cache — serve instantly, no API call
    logCache("hit", "userContext", "fresh cache");
    verboseApiResult("peer.context() -> representation (cached)", cachedContext?.representation);
    verboseList("peer.context() -> peerCard (cached)", cachedContext?.peerCard);

    serveContext(config.peerName, cachedContext, true, sessionLink, dedupSessionId);
    process.exit(0);
  }

  // Cache is stale or threshold reached — try a fresh fetch with timeout
  logCache("miss", "userContext", forceRefresh ? "threshold refresh" : "stale cache");

  const fetchResult = await Promise.race([
    fetchFreshContext(config, prompt).then(r => ({ ok: true as const, ...r })),
    new Promise<{ ok: false }>(resolve => setTimeout(() => resolve({ ok: false }), FETCH_TIMEOUT_MS)),
  ]).catch((): { ok: false } => ({ ok: false }));

  if (fetchResult.ok) {
    const { context } = fetchResult;
    if (forceRefresh) {
      markKnowledgeGraphRefreshed();
    }
    if (context) {
      serveContext(config.peerName, context, false, sessionLink, dedupSessionId);
      process.exit(0);
    }
  }

  // Fetch failed or timed out — silently fall back to stale cache
  const staleContext = getStaleCachedUserContext();
  if (staleContext) {
    logHook("user-prompt", "Serving stale cache after timeout");
    serveContext(config.peerName, staleContext, true, sessionLink, dedupSessionId);
  }
  // No cache at all — exit silently, context will arrive after session-start completes

  process.exit(0);
}

/**
 * Format and output context injection to Claude.
 */
function serveContext(
  peerName: string,
  context: any,
  cached: boolean,
  sessionLink?: string,
  dedupSessionId?: string,
): void {
  const { parts: contextParts } = formatCachedContext(context, peerName, dedupSessionId);
  if (contextParts.length === 0) return;

  const visMsg = visContextLine("user-prompt", { cached });
  outputContext(peerName, contextParts, sessionLink ? `${sessionLink}\n${visMsg}` : visMsg);
}

async function fetchFreshContext(config: any, prompt: string): Promise<{ context: any }> {
  const honcho = new Honcho(getHonchoClientOptions(config));
  const observationMode = getObservationMode(config);

  // unified: user self-observations — query via userPeer (no target).
  // directional: ai cross-observations — query via aiPeer with target.
  const contextPeer = observationMode === "unified"
    ? await honcho.peer(config.peerName)
    : await honcho.peer(config.aiPeer);
  const contextTarget = observationMode === "unified" ? undefined : config.peerName;
  const contextLabel = observationMode === "unified" ? "userPeer.context" : "aiPeer.context";

  const startTime = Date.now();

  // Try search-based context first — returns conclusions relevant to the prompt.
  // Fall back to the raw prompt (truncated) when no high-signal topics match:
  // natural text embeds well, and it keeps non-English prompts working.
  const topics = extractTopics(prompt);
  const searchQuery = topics.length > 0 ? topics.join(" ") : prompt.trim().slice(0, 300);

  let contextResult: any = null;

  if (searchQuery) {
    try {
      // The representation merges search hits with frequent/recent conclusions
      // into one timestamp-ordered string, so prompt-relevant lines get lost.
      // Query matched conclusions separately and let the formatter put them first.
      const conclusionScope = contextTarget
        ? contextPeer.conclusionsOf(contextTarget)
        : contextPeer.conclusions;
      const [ctx, matched] = await Promise.all([
        contextPeer.context({
          ...(contextTarget ? { target: contextTarget } : {}),
          searchQuery,
          searchTopK: 5,
          searchMaxDistance: 0.7,
          maxConclusions: 15,
          includeMostFrequent: true,
        }),
        conclusionScope.query(searchQuery, 5).catch((): any[] => []),
      ]);
      contextResult = ctx;
      if (contextResult && matched?.length) {
        contextResult.searchMatched = matched.map((c: any) => c.content).filter(Boolean);
      }
      logApiCall(contextLabel, "GET", `search: ${searchQuery.slice(0, 60)}`, Date.now() - startTime, true);
    } catch (e) {
      // Search failed — fall through to static context
      logHook("user-prompt", `Search context failed, falling back to static: ${e}`);
    }
  }

  // Fallback: static context (no search query)
  if (!contextResult) {
    contextResult = await contextPeer.context({
      ...(contextTarget ? { target: contextTarget } : {}),
      maxConclusions: 15,
      includeMostFrequent: true,
    });
    logApiCall(contextLabel, "GET", `static context`, Date.now() - startTime, true);
  }

  if (contextResult) {
    setCachedUserContext(contextResult);
    verboseApiResult("peer.context() -> representation (fresh)", (contextResult as any).representation);
    verboseList("peer.context() -> peerCard (fresh)", (contextResult as any).peerCard);
  }

  return { context: contextResult };
}

export function stripConclusionLine(line: string): string {
  return line.replace(/^\[.*?\]\s*/, "").replace(/^- /, "").trim();
}

export function formatCachedContext(context: any, peerName: string, dedupSessionId?: string): { parts: string[]; conclusionCount: number } {
  const parts: string[] = [];
  let conclusionCount = 0;
  let totalDropped = 0;
  const rep = context?.representation;

  // Cross-turn dedup (#40): one shared hash set, seeded from what this session has
  // already injected, so a conclusion never repeats turn after turn. Conclusions
  // and the Profile below share it and we record() once at the end — recording per
  // block would double-increment the session turn counter.
  const alreadyHashed = dedupSessionId ? getInjectedHashesForSession(dedupSessionId) : null;
  const freshHashes: string[] = [];

  // Within-turn ordering (#63): prompt-matched conclusions first (semantic search),
  // then the newest representation lines. The representation is oldest-first, so
  // taking its head would inject the stalest facts.
  const seen = new Set<string>();
  const accept = (text: string, cap: number, out: string[]): void => {
    const clean = stripConclusionLine(text);
    if (!clean || out.length >= cap) return;
    const key = clean.toLowerCase();
    if (seen.has(key)) return;
    const hash = hashLine(normalizeLine(text));
    if (alreadyHashed?.has(hash)) {
      totalDropped = totalDropped + 1;
      return;
    }
    seen.add(key);
    out.push(clean);
    if (alreadyHashed) {
      alreadyHashed.add(hash);
      freshHashes.push(hash);
    }
  };

  const selected: string[] = [];
  for (const c of context?.searchMatched ?? []) accept(String(c), 5, selected);

  if (typeof rep === "string" && rep.trim()) {
    const lines = rep.split("\n").filter((l: string) => l.trim() && !l.startsWith("#"));
    // Sort newest-first when lines carry a leading [timestamp]; otherwise keep order.
    const stamped = lines.map((l: string, i: number) => ({ l, t: l.match(/^\[([^\]]+)\]/)?.[1] ?? "", i }));
    stamped.sort((a, b) => (b.t.localeCompare(a.t)) || (a.i - b.i));
    for (const { l } of stamped) accept(l, 5, selected);
  }

  if (selected.length > 0) {
    conclusionCount = selected.length;
    parts.push(`Relevant conclusions: ${selected.join("; ")}`);
  }

  // Profile (peer card): same dedup gate. On the first turn the whole card is
  // fresh; on later turns its lines are already hashed, so it collapses to nothing.
  const peerCard = context?.peerCard;
  if (peerCard?.length) {
    const selectedPeerCard: string[] = [];
    for (const line of peerCard) accept(String(line), peerCard.length, selectedPeerCard);
    if (selectedPeerCard.length > 0) {
      parts.push(`Profile: ${selectedPeerCard.join("; ")}`);
    }
  }

  if (dedupSessionId && freshHashes.length > 0) {
    recordInjectedHashes(dedupSessionId, freshHashes);
  }

  if (totalDropped > 0 && isLoggingEnabled()) {
    logHook("user-prompt", `Dedup dropped ${totalDropped} already-injected line(s) for session ${dedupSessionId}`);
  }

  return { parts, conclusionCount };
}

function outputContext(peerName: string, contextParts: string[], systemMsg?: string): void {
  let output: any = {
    hookSpecificOutput: {
      hookEventName: "UserPromptSubmit",
      additionalContext: `[Honcho Memory for ${peerName}]: ${contextParts.join(" | ")}`,
    },
  };
  if (systemMsg) {
    output = addSystemMessage(output, systemMsg);
  }
  console.log(JSON.stringify(output));
}
