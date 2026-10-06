/**
 * FlowDeck — OpenCode v2 plugin entrypoint.
 *
 * Ported from the OpenCode v1 plugin API (`@opencode-ai/plugin` function
 * entrypoints returning hooks) to the v2 API (`Plugin.define` + `setup(ctx)`),
 * per https://opencode.ai/v2/docs/build/plugins/migrate-v1:
 *
 *   v1 `config` hook      → per-domain transforms (agent / mcp / command /
 *                           skill) + a `session.hook("context")` for rules
 *   v1 `tool` map         → `ctx.tool.transform(editor.add(...))`
 *   v1 tool.execute.*     → `ctx.tool.hook("execute.before"/"execute.after")`
 *   v1 `event` callback   → `ctx.event.subscribe()` async iteration
 *   v1 `client.app.log`   → console logging (v2 context has no log domain)
 *
 * Tool modules keep their V1 shape through `src/compat/tool.ts` and are
 * converted to V2 definitions at registration time.
 */
import { Plugin } from "@opencode/plugin"
import { existsSync, readFileSync, readdirSync } from "fs"
import { basename, dirname, join } from "path"
import { fileURLToPath } from "url"
import { z } from "zod"

import {
  buildSelectionDiagnostics,
  detectProjectLanguages,
  getStartupRulePaths,
  selectRulePaths,
} from "./services/lazy-rule-loader"
import { LoopDetector } from "./services/loop-detector"

import { getAgentConfigs } from "./agents/index"
import { loadFlowDeckConfig, resolveAgentModels, type FlowDeckConfig } from "./config/index"
import type { AgentConfig } from "./agents/types"
import { sessionStartHook } from "./hooks/session-start"
import { sessionEventsHook } from "./hooks/session-events"
import { toolGuardHook } from "./hooks/tool-guard"
import { buildFlowDeckMcpsWithMeta } from "./mcp/index"
import type { LocalMcp, RemoteMcp } from "./mcp/index"
import { captureLessonTool, reviewLessonsTool } from "./tools/capture-lesson"
import { codebaseStateTool } from "./tools/codebase-state"
import { fdxValidateTool } from "./tools/fdx-validate"
import { fdxWorktreeTool } from "./tools/fdx-worktree"
import {
  fdxBatchTool,
  fdxContextTool,
  fdxDecisionsTool,
  fdxDiffTool,
  fdxGitTool,
  fdxGraphTool,
  fdxGrepTool,
  fdxImpactTool,
  fdxLintTool,
  fdxLsTool,
  fdxOutlineTool,
  fdxReadTool,
  fdxSearchTool,
  fdxTestTool,
  fdxTreeTool,
} from "./tools/fdx"
import type { ToolContext, ToolResult, ToolDefinition } from "./compat/tool"
import { hashEditTool } from "./tools/hash-edit"
import { loadRulesTool, listRulesTool } from "./tools/load-rules"
import { planningStateTool } from "./tools/planning-state"
import { repoMemoryTool } from "./tools/repo-memory"

const __dir = dirname(fileURLToPath(import.meta.url))

/** FlowDeck plugin id — scopes plugin storage and identifies it in diagnostics. */
export const PLUGIN_ID = "flowdeck"

/** All tools registered by FlowDeck, keyed by tool name. */
export const FLOWDECK_TOOLS: Record<string, ToolDefinition> = {
  "planning-state": planningStateTool,
  "codebase-state": codebaseStateTool,
  "repo-memory": repoMemoryTool,
  "hash-edit": hashEditTool,
  "load-rules": loadRulesTool,
  "list-rules": listRulesTool,
  "capture-lesson": captureLessonTool,
  "review-lessons": reviewLessonsTool,
  "fdx-context": fdxContextTool,
  "fdx-decisions": fdxDecisionsTool,
  "fdx-validate": fdxValidateTool,
  "fdx-worktree": fdxWorktreeTool,
  "fdx-read": fdxReadTool,
  "fdx-search": fdxSearchTool,
  "fdx-grep": fdxGrepTool,
  "fdx-batch": fdxBatchTool,
  "fdx-graph": fdxGraphTool,
  "fdx-impact": fdxImpactTool,
  "fdx-outline": fdxOutlineTool,
  "fdx-diff": fdxDiffTool,
  "fdx-git": fdxGitTool,
  "fdx-ls": fdxLsTool,
  "fdx-tree": fdxTreeTool,
  "fdx-test": fdxTestTool,
  "fdx-lint": fdxLintTool,
}

/** Select FlowDeck rule paths for injection into model context. */
function lazyLoadRulePaths(projectRoot: string): { paths: string[]; diagnostics: string } {
  const rulesDir = join(__dir, "..", "src", "rules")
  if (!existsSync(rulesDir)) return { paths: [], diagnostics: "[LazyRuleLoader] rules directory not found" }
  const detected = detectProjectLanguages(projectRoot)
  const paths = getStartupRulePaths(rulesDir, detected)
  const selection = selectRulePaths(rulesDir, { languages: detected, projectRoot })
  return { paths, diagnostics: buildSelectionDiagnostics(selection, { languages: detected, projectRoot }) }
}

/** Load FlowDeck slash commands from src/commands/*.md (parses frontmatter description). */
export function loadCommands(): Record<string, { description?: string; template: string }> {
  const dir = join(__dir, "..", "src", "commands")
  if (!existsSync(dir)) return {}
  const out: Record<string, { description?: string; template: string }> = {}
  try {
    for (const file of readdirSync(dir)) {
      if (!file.endsWith(".md")) continue
      const raw = readFileSync(join(dir, file), "utf-8")
      const fm = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/)
      const template = fm ? fm[2].trim() : raw
      const desc = fm?.[1].match(/^description:\s*(.+)$/m)?.[1].trim()
      out[basename(file, ".md")] = desc ? { description: desc, template } : { template }
    }
  } catch { /* ignore */ }
  return out
}

/**
 * Substitute command arguments into a command template, matching the v1
 * markdown command behavior: `$ARGUMENTS` is replaced in place; when the
 * template has no placeholder, arguments are appended as a new paragraph.
 */
export function substituteArguments(template: string, args: string): string {
  const trimmed = args.trim()
  if (template.includes("$ARGUMENTS")) return template.replaceAll("$ARGUMENTS", trimmed)
  return trimmed ? `${template}\n\n${trimmed}` : template
}

/** JSON Schema for a tool's zod `args` shape, as required by V2 tool registration. */
export function toolInputSchema(args: Parameters<typeof z.object>[0]) {
  return z.toJSONSchema(z.object(args), { io: "input" })
}

/** V2 tool result content types (`Tool.TextContent` / `Tool.FileContent`). */
export type V2FileContent = { type: "file"; uri: string; mime: string; name?: string }
export type V2ToolResult = {
  title?: string
  metadata?: Record<string, any>
  content: string | Array<{ type: "text"; text: string } | V2FileContent>
}

/**
 * Convert a V1 tool result (plain string or `{ title, output, metadata,
 * attachments }`) into the V2 `Tool.Result` shape.
 */
export function toV2Result(result: ToolResult): V2ToolResult {
  if (typeof result === "string") return { content: result }
  const content: Array<{ type: "text"; text: string } | V2FileContent> = [
    { type: "text", text: result.output },
  ]
  for (const attachment of result.attachments ?? []) {
    content.push({ type: "file", uri: attachment.url, mime: attachment.mime, name: attachment.filename })
  }
  return {
    ...(result.title === undefined ? {} : { title: result.title }),
    ...(result.metadata === undefined ? {} : { metadata: result.metadata }),
    content,
  }
}

/**
 * Convert FlowDeck's V1 MCP configs (`enabled`/`oauth` fields) to the V2
 * `Mcp.ServerConfig` shape (absence of `disabled` means enabled).
 */
export function convertMcpConfig(server: RemoteMcp | LocalMcp): Record<string, unknown> {
  if (server.type === "remote") {
    return {
      type: "remote",
      url: server.url,
      ...(server.headers ? { headers: server.headers } : {}),
      oauth: false,
    }
  }
  return {
    type: "local",
    command: server.command,
    ...(server.environment ? { environment: server.environment } : {}),
  }
}

/** Parse a `provider/model[#variant]` reference into the V2 `Model.Ref` shape. */
export function parseModelRef(model: string): { providerID: string; id: string; variant?: string } {
  const [providerID = "", rest = ""] = model.split("/")
  const [id = "", variant] = rest.split("#")
  return variant ? { providerID, id, variant } : { providerID, id }
}

/**
 * Fields to set on a V2 `Agent.Info` for one FlowDeck agent config. Exported
 * as a plain object so the mapping is testable without a live registry.
 */
export function agentEdits(cfg: AgentConfig): {
  description?: string
  system?: string
  mode: "subagent" | "primary" | "all"
  hidden: boolean
  color?: string
  steps?: number
  model?: { providerID: string; id: string; variant?: string }
  request: { body: Record<string, unknown> }
} {
  return {
    ...(cfg.description === undefined ? {} : { description: cfg.description }),
    ...(cfg.prompt === undefined ? {} : { system: cfg.prompt }),
    mode: cfg.mode ?? "subagent",
    hidden: cfg.hidden ?? false,
    ...(cfg.color === undefined ? {} : { color: cfg.color }),
    ...(cfg.steps === undefined ? {} : { steps: cfg.steps }),
    ...(cfg.model ? { model: parseModelRef(cfg.model) } : {}),
    request: {
      body: {
        ...(cfg.temperature === undefined ? {} : { temperature: cfg.temperature }),
      },
    },
  }
}

/** Parsed skill file (frontmatter + body) ready for V2 skill registration. */
export interface SkillInfoInput {
  id: string
  name: string
  description: string
  path: string
  content: string
}

/** Load FlowDeck skills from a directory of `<id>/SKILL.md` folders. */
export function loadSkillInfos(skillsDir: string): SkillInfoInput[] {
  if (!existsSync(skillsDir)) return []
  const out: SkillInfoInput[] = []
  for (const entry of readdirSync(skillsDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue
    const skillPath = join(skillsDir, entry.name, "SKILL.md")
    if (!existsSync(skillPath)) continue
    const raw = readFileSync(skillPath, "utf-8")
    const fm = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/)
    const name = fm?.[1].match(/^name:\s*(.+)$/m)?.[1].trim() ?? entry.name
    const description = fm?.[1].match(/^description:\s*(.+)$/m)?.[1].trim() ?? ""
    out.push({
      id: entry.name,
      name,
      description,
      path: skillPath,
      content: (fm ? fm[2] : raw).trim(),
    })
  }
  return out
}

/** Read `sessionID` out of a V2 event payload without relying on union narrowing. */
export function eventSessionID(event: { data?: unknown }): string {
  const data = event?.data
  if (typeof data === "object" && data !== null && "sessionID" in data) {
    const sessionID = (data as { sessionID?: unknown }).sessionID
    return typeof sessionID === "string" ? sessionID : ""
  }
  return ""
}

const plugin = Plugin.define({
  id: PLUGIN_ID,
  async setup(ctx) {
    const directory = ctx.location.directory

    // The V2 plugin context has no log domain (v1 used `client.app.log`);
    // console output is captured in the server log instead.
    const appLog = (msg: string): Promise<void> => {
      console.log(`[flowdeck] ${msg}`)
      return Promise.resolve()
    }

    let flowdeckConfig: FlowDeckConfig = loadFlowDeckConfig(directory)
    const loopDetector = new LoopDetector(flowdeckConfig.governance?.loopDetection, appLog)

    const { mcps } = buildFlowDeckMcpsWithMeta()

    // --- Agents (v1 `config` hook injected them into the agent map) ---
    await ctx.agent.transform((editor) => {
      const resolved = getAgentConfigs(resolveAgentModels(flowdeckConfig))
      for (const [name, cfg] of Object.entries(resolved)) {
        // `editor.update` creates the agent (seeded from Agent.Info.default) when absent.
        editor.update(name, (agent) => {
          const edits = agentEdits(cfg)
          if (edits.description !== undefined) agent.description = edits.description
          if (edits.system !== undefined) agent.system = edits.system
          if (edits.color !== undefined) agent.color = edits.color
          if (edits.steps !== undefined) agent.steps = edits.steps
          if (edits.model !== undefined) {
            // `Agent.Info.model` is a decoded Model.Ref (branded strings);
            // flowdeck.json stores "provider/model" strings, so parse them
            // here. The registry stores plain objects, so this cast only
            // bridges the schema's phantom brand types.
            agent.model = edits.model as unknown as typeof agent.model
          }
          agent.request.body.temperature = edits.request.body.temperature
          agent.mode = edits.mode
          agent.hidden = edits.hidden
        })
      }
      editor.default("orchestrator")
    })

    // --- MCP servers (v1 `config` hook merged them into cfg.mcp) ---
    await ctx.mcp.transform((editor) => {
      for (const [name, server] of Object.entries(mcps)) {
        editor.set(name, convertMcpConfig(server) as Parameters<typeof editor.set>[1])
      }
    })

    // --- Slash commands (v1 `config` hook injected markdown commands) ---
    await ctx.command.transform((editor) => {
      for (const [name, cmd] of Object.entries(loadCommands())) {
        editor.add({
          name,
          ...(cmd.description === undefined ? {} : { description: cmd.description }),
          execute: async ({ sessionID, prompt, delivery }) => {
            const text = substituteArguments(cmd.template, prompt.text)
            await ctx.session.prompt({ sessionID, text, delivery })
          },
        })
      }
    })

    // --- Skills (v1 `config` hook appended the skills directory to cfg.skills) ---
    await ctx.skill.transform((editor) => {
      for (const skill of loadSkillInfos(join(__dir, "..", "src", "skills"))) {
        // Skill.Info brands `id`/`path`; our loader produces plain strings,
        // which is what the registry stores at runtime.
        editor.add(skill as unknown as Parameters<typeof editor.add>[0])
      }
    })

    // --- Language rules (v1 `config` hook appended paths to cfg.instructions) ---
    const { paths: rulePaths, diagnostics } = lazyLoadRulePaths(directory)
    appLog(diagnostics)
    const ruleTexts: string[] = []
    for (const p of rulePaths) {
      try {
        ruleTexts.push(readFileSync(p, "utf-8"))
      } catch { /* ignore unreadable rule file */ }
    }
    if (ruleTexts.length > 0) {
      await ctx.session.hook("context", (event) => {
        for (const text of ruleTexts) {
          event.system.push({ type: "text", text })
        }
      })
    }

    // --- Tools (v1 `tool` map) ---
    await ctx.tool.transform((editor) => {
      for (const [name, def] of Object.entries(FLOWDECK_TOOLS)) {
        editor.add({
          name,
          description: def.description,
          input: toolInputSchema(def.args),
          execute: async (input, toolCtx) => {
            const v1Context: ToolContext = {
              directory,
              worktree: directory,
              abort: toolCtx.signal,
              sessionID: toolCtx.sessionID,
              messageID: toolCtx.messageID,
              agent: toolCtx.agent,
              metadata: (meta) => {
                void toolCtx.progress({
                  ...(meta.title === undefined ? {} : { title: meta.title }),
                  ...(meta.metadata ?? {}),
                })
              },
              ask: async () => {},
            }
            const result = await def.execute(input as never, v1Context)
            return toV2Result(result)
          },
        })
      }
    })

    // --- Tool guard + loop detection (v1 tool.execute.before/after) ---
    await ctx.tool.hook("execute.before", async (event) => {
      const args = (event.input ?? {}) as Record<string, unknown>
      await toolGuardHook(
        { directory, agent: event.agent },
        { tool: event.tool, sessionID: event.sessionID, agent: event.agent, args },
        { args },
      )
      const loop = loopDetector.checkBefore(event.tool, args, event.sessionID)
      if (loop.action === "block") throw new Error(loop.escalationMessage)
      if (loop.action === "warn") appLog(loop.message)
    })

    await ctx.tool.hook("execute.after", async (event) => {
      appLog(`[tool] done tool=${event.tool} session=${event.sessionID}`)
      // V1 only exposed the input here; keep call-count tracking working by
      // recording the structured result when available.
      const output =
        event.status === "completed"
          ? JSON.stringify(event.result ?? "")
          : `error: ${event.error.message}`
      loopDetector.recordAfter(event.tool, (event.input ?? {}) as Record<string, unknown>, output, event.sessionID)
    })

    // --- Session events (v1 `event` callback) ---
    const controller = new AbortController()
    void (async () => {
      try {
        for await (const event of ctx.event.subscribe({ signal: controller.signal })) {
          const type: string = event?.type ?? ""
          if (type === "session.created") {
            await sessionStartHook({ directory }, appLog)
          } else if (type === "session.idle" || type === "session.execution.succeeded") {
            // v1 fired checkpoints on session.idle; v2.0.24 does not emit
            // session.idle on the public event stream — the execution
            // lifecycle events carry the same sessionID and fire per turn.
            await sessionEventsHook({ directory }, "idle", eventSessionID(event))
          } else if (type === "session.error" || type === "session.execution.failed") {
            await sessionEventsHook({ directory }, "error", eventSessionID(event))
          }
        }
      } catch (error) {
        if (!controller.signal.aborted) appLog(`event stream error: ${error}`)
      }
    })()

    return () => controller.abort()
  },
})

export default plugin