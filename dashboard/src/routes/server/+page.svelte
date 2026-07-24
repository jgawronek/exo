<script lang="ts">
  import { browser } from "$app/environment";
  import { fade } from "svelte/transition";
  import HeaderNav from "$lib/components/HeaderNav.svelte";
  import IntegrationCard from "$lib/components/IntegrationCard.svelte";
  import {
    instances,
    masterNodeId,
    nodeIdentities,
    nodeNetwork,
    refreshState,
  } from "$lib/stores/app.svelte";
  import { onMount } from "svelte";

  const fallbackApiUrl = browser
    ? window.location.origin.replace("localhost", "127.0.0.1")
    : "http://127.0.0.1:52415";
  const apiPort = browser && window.location.port ? window.location.port : "52415";

  const instancesData = $derived(instances());

  let modelCapabilities = $state<Record<string, string[]>>({});
  let modelContextLengths = $state<Record<string, number>>({});
  let modelReasoningDialects = $state<Record<string, string>>({});

  interface RunningInstance {
    instanceId: string;
    shortId: string;
    modelId: string;
    alias: string;
    nodeIds: string[];
  }

  const runningInstances = $derived.by(() => {
    const result: RunningInstance[] = [];
    for (const [instanceId, wrapper] of Object.entries(instancesData)) {
      if (!wrapper || typeof wrapper !== "object") continue;
      const values = Object.values(wrapper as Record<string, unknown>);
      if (values.length === 0) continue;
      const inst = values[0] as {
        shardAssignments?: {
          modelId?: string;
          nodeToRunner?: Record<string, string>;
        };
      };
      const modelId = inst?.shardAssignments?.modelId;
      if (!modelId) continue;
      const shortId = instanceId.slice(0, 8);
      result.push({
        instanceId,
        shortId,
        modelId,
        alias: `${modelId}@${shortId}`,
        nodeIds: Object.keys(inst.shardAssignments?.nodeToRunner ?? {}),
      });
    }
    return result;
  });

  let selectedInstanceId = $state<string | null>(null);
  $effect(() => {
    if (runningInstances.length === 0) {
      selectedInstanceId = null;
      return;
    }
    if (
      !runningInstances.some((inst) => inst.instanceId === selectedInstanceId)
    ) {
      selectedInstanceId = runningInstances[0].instanceId;
    }
  });

  const selectedInstance = $derived(
    runningInstances.find((inst) => inst.instanceId === selectedInstanceId) ??
      null,
  );
  const selectedAlias = $derived(selectedInstance?.alias ?? "your-model-id");
  const instanceAliases = $derived(runningInstances.map((inst) => inst.alias));

  function nodeName(nodeId: string): string {
    return nodeIdentities()[nodeId]?.friendlyName || nodeId.slice(0, 8);
  }

  // Prefer the elected master's LAN IP so configs work from other machines,
  // not just the one the dashboard happens to be open on.
  function isUsableIpv4(address: string): boolean {
    return (
      /^\d+\.\d+\.\d+\.\d+$/.test(address) &&
      !address.startsWith("127.") &&
      !address.startsWith("169.254.")
    );
  }

  function nodeLanIp(nodeId: string): string | null {
    const network = nodeNetwork()[nodeId];
    // Wired interfaces first: ethernet IPs are the most reliable targets
    // for API traffic when a node has both wired and wifi addresses.
    const candidates: { address: string; wired: boolean }[] = [];
    for (const iface of network?.interfaces ?? []) {
      const wired = iface.interfaceType === "ethernet";
      const push = (address: unknown) => {
        if (typeof address === "string") candidates.push({ address, wired });
      };
      push(iface.ipAddress);
      push(iface.ipv4);
      for (const addr of iface.addresses ?? []) {
        push(typeof addr === "string" ? addr : addr?.address);
      }
      for (const addr of iface.ipAddresses ?? []) push(addr);
      for (const addr of iface.ips ?? []) push(addr);
    }
    const usable = candidates.filter((c) => isUsableIpv4(c.address));
    const isLan = (address: string) =>
      address.startsWith("192.168.") ||
      address.startsWith("10.") ||
      /^172\.(1[6-9]|2\d|3[01])\./.test(address);
    const pick =
      usable.find((c) => c.wired && isLan(c.address)) ??
      usable.find((c) => isLan(c.address)) ??
      usable[0];
    return pick?.address ?? null;
  }

  const masterIp = $derived.by(() => {
    const id = masterNodeId();
    return id ? nodeLanIp(id) : null;
  });

  /** First node of the selected instance that has a usable LAN IP. */
  const selectedInstanceEndpoint = $derived.by(() => {
    for (const nodeId of selectedInstance?.nodeIds ?? []) {
      const ip = nodeLanIp(nodeId);
      if (ip) return { ip, nodeId };
    }
    return null;
  });

  const apiUrl = $derived(
    selectedInstanceEndpoint
      ? `http://${selectedInstanceEndpoint.ip}:${apiPort}`
      : masterIp
        ? `http://${masterIp}:${apiPort}`
        : fallbackApiUrl,
  );

  const apiUrlSourceLabel = $derived(
    selectedInstanceEndpoint
      ? `instance node: ${nodeName(selectedInstanceEndpoint.nodeId)}`
      : masterIp
        ? "master node"
        : null,
  );

  // Capability lookups accept either a plain model id or an instance alias
  // (model@shortid); alias entries come from /v1/models but fall back to the
  // base model's card if the fetch predates the instance.
  function capsOf(idOrAlias: string): string[] {
    return (
      modelCapabilities[idOrAlias] ??
      modelCapabilities[idOrAlias.split("@")[0]] ??
      []
    );
  }
  function contextLengthOf(idOrAlias: string): number {
    return (
      modelContextLengths[idOrAlias] ??
      modelContextLengths[idOrAlias.split("@")[0]] ??
      0
    );
  }
  function dialectOf(idOrAlias: string): string | undefined {
    return (
      modelReasoningDialects[idOrAlias] ??
      modelReasoningDialects[idOrAlias.split("@")[0]]
    );
  }

  let opusModel = $state("");
  let sonnetModel = $state("");
  let haikuModel = $state("");
  let codexModel = $state("");
  let codexMcpPath = $state("/Users/username");
  let openClawModel = $state("");
  let piModel = $state("");

  $effect(() => {
    opusModel = selectedAlias;
    sonnetModel = selectedAlias;
    haikuModel = selectedAlias;
    codexModel = selectedAlias;
    openClawModel = selectedAlias;
    piModel = selectedAlias;
  });

  const claudeShellCommand = $derived(
    [
      `ANTHROPIC_BASE_URL=${apiUrl} \\`,
      `ANTHROPIC_API_KEY=x \\`,
      `ANTHROPIC_DEFAULT_OPUS_MODEL=${opusModel} \\`,
      `ANTHROPIC_DEFAULT_SONNET_MODEL=${sonnetModel} \\`,
      `ANTHROPIC_DEFAULT_HAIKU_MODEL=${haikuModel} \\`,
      `API_TIMEOUT_MS=3000000 \\`,
      `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \\`,
      `claude`,
    ].join("\n"),
  );

  const claudeSettingsJson = $derived(
    JSON.stringify(
      {
        env: {
          ANTHROPIC_BASE_URL: apiUrl,
          ANTHROPIC_API_KEY: "x",
          ANTHROPIC_DEFAULT_OPUS_MODEL: opusModel,
          ANTHROPIC_DEFAULT_SONNET_MODEL: sonnetModel,
          ANTHROPIC_DEFAULT_HAIKU_MODEL: haikuModel,
          API_TIMEOUT_MS: "3000000",
          CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1",
        },
      },
      null,
      2,
    ),
  );

  const openCodeConfig = $derived.by(() => {
    const models: Record<string, Record<string, unknown>> = {};
    for (const instance of runningInstances) {
      const caps = capsOf(instance.alias);
      const ctxLen = contextLengthOf(instance.alias);
      const dialect = dialectOf(instance.alias);
      const entry: Record<string, unknown> = { name: instance.alias };
      if (ctxLen > 0) {
        entry.limit = { context: ctxLen, output: Math.min(ctxLen, 16384) };
      }
      if (caps.includes("vision")) {
        entry.modalities = { input: ["text", "image"], output: ["text"] };
      }
      // Reasoning round-trip: opencode's `interleaved` field tells the
      // openai-compatible adapter to send the assistant's prior
      // reasoning_content back in subsequent turns. Emit it for dialects
      // whose chat templates use prior reasoning:
      //   - `tool_conditional` (DeepSeek V3.2 / V4): wrapper preserves all
      //     reasoning when tools are present.
      //   - `post_last_user` (Qwen3-Thinking, GLM 4.5+, MiniMax M2.x):
      //     Jinja template reads reasoning_content for assistant turns since
      //     the last user message — exactly the tool-chain window.
      //   - `channel` (gpt-oss / Harmony): the model's Jinja template reads
      //     `message.thinking` rather than `message.reasoning_content`, but
      //     the server bridges `reasoning_content` → `thinking` before
      //     rendering, so the round-trip works through the standard field.
      // `suffix` (Kimi): reasoning lives in content; no separate field path.
      if (
        dialect === "tool_conditional" ||
        dialect === "post_last_user" ||
        dialect === "channel"
      ) {
        entry.interleaved = { field: "reasoning_content" };
      }
      models[instance.alias] = entry;
    }
    if (Object.keys(models).length === 0) {
      models["your-model-id"] = { name: "your-model-name" };
    }
    return JSON.stringify(
      {
        $schema: "https://opencode.ai/config.json",
        provider: {
          exo: {
            npm: "@ai-sdk/openai-compatible",
            name: "XEO",
            options: {
              baseURL: `${apiUrl}/v1`,
              apiKey: "x",
            },
            models,
          },
        },
        model: `exo/${selectedAlias}`,
      },
      null,
      2,
    );
  });

  const codexShellCommand = $derived(`XEO_API_KEY=x npx @openai/codex`);

  const codexConfig = $derived(
    [
      `model = "${codexModel}"`,
      `model_provider = "exo"`,
      ``,
      `[model_providers.exo]`,
      `name = "XEO"`,
      `base_url = "${apiUrl}/v1"`,
      `env_key = "XEO_API_KEY"`,
      ``,
      `[mcp_servers.filesystem]`,
      `command = "npx"`,
      `args = ["-y", "@modelcontextprotocol/server-filesystem", "${codexMcpPath}"]`,
    ].join("\n"),
  );

  const openClawConfig = $derived(
    JSON.stringify(
      {
        gateway: { mode: "local" },
        models: {
          providers: {
            exo: {
              baseUrl: `${apiUrl}/v1`,
              apiKey: "x",
              api: "openai-completions",
              models: [
                {
                  id: openClawModel,
                  name: "XEO local",
                  input: capsOf(openClawModel).includes("vision")
                    ? ["text", "image"]
                    : ["text"],
                },
              ],
            },
          },
        },
        agents: {
          defaults: {
            model: `exo/${openClawModel}`,
          },
        },
      },
      null,
      2,
    ),
  );

  const piModelsJson = $derived.by(() => {
    const models: Record<string, unknown>[] = [];
    for (const instance of runningInstances) {
      const caps = capsOf(instance.alias);
      const ctxLen = contextLengthOf(instance.alias);
      const entry: Record<string, unknown> = { id: instance.alias };
      if (caps.includes("vision")) {
        entry.input = ["text", "image"];
      }
      // Mark thinking-capable models so pi surfaces its thinking-level selector
      // for them. exo capability strings: "thinking" (model emits reasoning
      // content) and "thinking_toggle" (user can turn it on/off).
      if (caps.includes("thinking") || caps.includes("thinking_toggle")) {
        entry.reasoning = true;
      }
      if (ctxLen > 0) {
        entry.contextWindow = ctxLen;
      }
      models.push(entry);
    }
    if (models.length === 0) {
      models.push({ id: "your-model-id" });
    }
    return JSON.stringify(
      {
        providers: {
          exo: {
            baseUrl: `${apiUrl}/v1`,
            api: "openai-completions",
            apiKey: "exo",
            compat: {
              supportsDeveloperRole: false,
              // exo's OpenAI surface takes a boolean `enable_thinking` toggle,
              // not graded effort levels, so disable pi's `reasoning_effort`
              // parameter and use the matching top-level-boolean format.
              supportsReasoningEffort: false,
              thinkingFormat: "qwen",
            },
            models,
          },
        },
      },
      null,
      2,
    );
  });

  const piShellCommand = $derived(`pi --provider exo --model ${piModel}`);

  const ollamaCommand = $derived(
    `OLLAMA_HOST=${apiUrl}/ollama ollama run ${selectedAlias}`,
  );

  const openWebUiCommand = $derived(
    [
      `docker run -d -p 3000:8080 \\`,
      `  -e OLLAMA_BASE_URL=${apiUrl.replace("localhost", "host.docker.internal")}/ollama \\`,
      `  -v open-webui:/app/backend/data \\`,
      `  --name open-webui \\`,
      `  ghcr.io/open-webui/open-webui:main`,
    ].join("\n"),
  );

  const n8nDockerCommand = $derived(
    [
      `docker run -d -p 5678:5678 \\`,
      `  -v n8n_data:/home/node/.n8n \\`,
      `  --name n8n \\`,
      `  docker.n8n.io/n8nio/n8n`,
    ].join("\n"),
  );

  const n8nCredentialSteps = $derived(
    [
      `1. Go to Credentials → Add Credential → search "OpenAI API"`,
      `2. Set API Key to: x`,
      `3. Set Base URL to: ${apiUrl.replace("127.0.0.1", "host.docker.internal").replace("localhost", "host.docker.internal")}/v1`,
      `4. Save the credential`,
    ].join("\n"),
  );

  const n8nWorkflowSteps = $derived(
    [
      `1. Create a new workflow → "Start from Scratch"`,
      `2. Add an "AI Agent" or "Basic LLM Chain" node`,
      `3. Inside it, add an "OpenAI Chat Model" sub-node`,
      `4. Select the OpenAI credential you just created`,
      `5. Set Model to "From list" and pick your instance (e.g. ${selectedAlias})`,
      `6. Optionally toggle "Use Responses API", add Built-in Tools, or click "Add Option" for sampling settings`,
      `7. Connect a "Chat Trigger" node for interactive chat`,
      `8. On the Chat Trigger, enable "Allow File Uploads" for vision`,
    ].join("\n"),
  );

  const firefoxConfig = $derived(
    [
      `1. Open about:config in Firefox`,
      `2. Set browser.ml.chat.enabled to true`,
      `3. Set browser.ml.chat.hideLocalhost to false`,
      `4. Set browser.ml.chat.provider to: ${apiUrl}/`,
    ].join("\n"),
  );

  const tabs = [
    "Claude Code",
    "OpenCode",
    "Codex",
    "OpenClaw",
    "Pi",
    "Open WebUI",
    "n8n",
    "Firefox",
  ] as const;
  type Tab = (typeof tabs)[number];
  const stored = browser ? localStorage.getItem("exo-integrations-tab") : null;
  let activeTab = $state<Tab>(
    stored && tabs.includes(stored as Tab) ? (stored as Tab) : "Claude Code",
  );
  $effect(() => {
    if (browser) localStorage.setItem("exo-integrations-tab", activeTab);
  });

  const selectClass =
    "bg-black/30 border border-xeo-light-gray/20 rounded px-2 py-1.5 text-white font-mono text-xs focus:border-xeo-green/50 focus:outline-none appearance-none cursor-pointer";

  onMount(async () => {
    refreshState();
    try {
      const resp = await fetch("/v1/models");
      const data = (await resp.json()) as {
        data: {
          id: string;
          capabilities: string[];
          context_length: number;
          reasoning_dialect?: string;
        }[];
      };
      const caps: Record<string, string[]> = {};
      const ctxs: Record<string, number> = {};
      const dialects: Record<string, string> = {};
      for (const model of data.data) {
        caps[model.id] = model.capabilities || [];
        if (model.context_length > 0) ctxs[model.id] = model.context_length;
        if (model.reasoning_dialect)
          dialects[model.id] = model.reasoning_dialect;
      }
      modelCapabilities = caps;
      modelContextLengths = ctxs;
      modelReasoningDialects = dialects;
    } catch {
      /* ignore */
    }
  });
</script>

<div class="min-h-screen bg-xeo-dark-gray flex flex-col">
  <HeaderNav showHome={true} />

  <main
    class="flex-1 max-w-3xl mx-auto w-full px-4 md:px-6 py-8"
    in:fade={{ duration: 200 }}
  >
    <div class="mb-8">
      <h1
        class="text-white text-xl md:text-2xl font-semibold tracking-wide mb-2"
      >
        Server
      </h1>
      <p class="text-xeo-light-gray/60 text-sm">
        Connect external tools to your XEO server.
      </p>
    </div>

    <!-- Available Instances -->
    <div class="mb-6">
      <span
        class="text-xeo-light-gray/70 text-xs uppercase tracking-wider block mb-2"
        >Available Instances</span
      >
      {#if runningInstances.length > 0}
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {#each runningInstances as instance (instance.instanceId)}
            <button
              onclick={() => (selectedInstanceId = instance.instanceId)}
              class="text-left rounded-md px-3 py-2.5 border transition-all cursor-pointer
                {selectedInstanceId === instance.instanceId
                ? 'bg-xeo-green/10 border-xeo-green/40'
                : 'bg-black/20 border-xeo-light-gray/10 hover:border-xeo-light-gray/30'}"
            >
              <div class="flex items-baseline justify-between gap-2">
                <span
                  class="text-sm font-medium truncate
                    {selectedInstanceId === instance.instanceId
                    ? 'text-xeo-green'
                    : 'text-white/90'}"
                >
                  {instance.modelId.split("/").pop()}
                </span>
                <span
                  class="text-[10px] font-mono text-xeo-light-gray/50 shrink-0"
                  >@{instance.shortId}</span
                >
              </div>
              <div
                class="text-[10px] font-mono text-xeo-light-gray/60 truncate mt-0.5"
              >
                {instance.alias}
              </div>
              <div class="text-[10px] text-xeo-light-gray/40 mt-1">
                {instance.nodeIds.length} node{instance.nodeIds.length === 1
                  ? ""
                  : "s"}: {instance.nodeIds.map(nodeName).join(", ")}
              </div>
            </button>
          {/each}
        </div>
        <p class="text-xeo-light-gray/40 text-[11px] mt-2">
          Configs below target the selected instance. Use the plain model id
          instead of the alias to load-balance across all instances of that
          model.
        </p>
      {:else}
        <p class="text-xeo-light-gray/40 text-xs italic">
          No instances currently running
        </p>
      {/if}
    </div>

    <!-- Status -->
    <div class="mb-8">
      <span class="text-xeo-light-gray/70 text-xs uppercase tracking-wider"
        >API Endpoint</span
      >
      <span class="text-white font-mono text-sm ml-2">{apiUrl}</span>
      {#if apiUrlSourceLabel}
        <span class="text-xeo-light-gray/40 text-[10px] ml-2 uppercase"
          >{apiUrlSourceLabel}</span
        >
      {/if}
    </div>

    <!-- API Endpoints -->
    <div class="mb-8">
      <div
        class="flex flex-col sm:flex-row gap-3 text-xs font-mono text-xeo-light-gray/70"
      >
        <div
          class="flex-1 bg-black/20 border border-xeo-light-gray/10 rounded px-3 py-2"
        >
          <span class="text-xeo-light-gray/40 text-[10px] uppercase block mb-1"
            >OpenAI-compatible</span
          >
          <span class="text-white/80">{apiUrl}/v1</span>
        </div>
        <div
          class="flex-1 bg-black/20 border border-xeo-light-gray/10 rounded px-3 py-2"
        >
          <span class="text-xeo-light-gray/40 text-[10px] uppercase block mb-1"
            >Claude-compatible</span
          >
          <span class="text-white/80">{apiUrl}</span>
        </div>
        <div
          class="flex-1 bg-black/20 border border-xeo-light-gray/10 rounded px-3 py-2"
        >
          <span class="text-xeo-light-gray/40 text-[10px] uppercase block mb-1"
            >Ollama-compatible</span
          >
          <span class="text-white/80">{apiUrl}/ollama</span>
        </div>
      </div>
    </div>

    <!-- Tabs -->
    <div
      class="flex flex-wrap gap-2 mb-6 border-b border-xeo-light-gray/10 pb-3"
    >
      {#each tabs as tab}
        <button
          onclick={() => (activeTab = tab)}
          class="px-3 py-1.5 text-xs rounded-md transition-all cursor-pointer
            {activeTab === tab
            ? 'bg-xeo-green/15 text-xeo-green border border-xeo-green/30'
            : 'text-xeo-light-gray/60 hover:text-white/80 border border-transparent hover:border-xeo-light-gray/20'}"
        >
          {tab}
        </button>
      {/each}
    </div>

    <!-- Tab Content -->
    <div class="space-y-4">
      {#if activeTab === "Claude Code"}
        {#if instanceAliases.length > 1}
          <div class="grid grid-cols-3 gap-3 text-xs">
            {#each [{ label: "Opus", bind: () => opusModel, set: (v: string) => (opusModel = v) }, { label: "Sonnet", bind: () => sonnetModel, set: (v: string) => (sonnetModel = v) }, { label: "Haiku", bind: () => haikuModel, set: (v: string) => (haikuModel = v) }] as tier}
              <div>
                <span
                  class="text-xeo-light-gray/50 text-[10px] uppercase tracking-wider block mb-1"
                  >{tier.label}</span
                >
                <select
                  value={tier.bind()}
                  onchange={(e) =>
                    tier.set((e.target as HTMLSelectElement).value)}
                  class="w-full {selectClass}"
                >
                  {#each instanceAliases as alias}
                    <option value={alias}>{alias.split("/").pop()}</option>
                  {/each}
                </select>
              </div>
            {/each}
          </div>
        {/if}
        <IntegrationCard
          title="Shell Command"
          subtitle="Run in terminal"
          description="Launch Claude Code with XEO as the backend. Paste this into your terminal."
          config={claudeShellCommand}
          language="bash"
        />
        <IntegrationCard
          title="Settings File"
          subtitle="~/.claude/settings.json"
          description="Or add this to your Claude Code settings for persistent configuration."
          config={claudeSettingsJson}
        />
      {:else if activeTab === "OpenCode"}
        <IntegrationCard
          title="Config File"
          subtitle="opencode.json"
          description="Add this to your project root or ~/.config/opencode/opencode.json for global config. Each running instance is listed as its own model. Vision models automatically get image input modality."
          config={openCodeConfig}
        />
      {:else if activeTab === "Codex"}
        <div class="flex gap-3 text-xs">
          {#if instanceAliases.length > 1}
            <div>
              <span
                class="text-xeo-light-gray/50 text-[10px] uppercase tracking-wider block mb-1"
                >Instance</span
              >
              <select bind:value={codexModel} class={selectClass}>
                {#each instanceAliases as alias}
                  <option value={alias}>{alias.split("/").pop()}</option>
                {/each}
              </select>
            </div>
          {/if}
          <div class="flex-1">
            <span
              class="text-xeo-light-gray/50 text-[10px] uppercase tracking-wider block mb-1"
              >MCP Filesystem Path</span
            >
            <input
              type="text"
              bind:value={codexMcpPath}
              class="w-full bg-black/30 border border-xeo-light-gray/20 rounded px-2 py-1.5 text-white font-mono text-xs focus:border-xeo-green/50 focus:outline-none"
            />
          </div>
        </div>
        <IntegrationCard
          title="Config File"
          subtitle="~/.codex/config.toml"
          description="Add this to your Codex CLI config so the model and provider persist."
          config={codexConfig}
        />
        <IntegrationCard
          title="Shell Command"
          subtitle="Run in terminal"
          description="Launch Codex with XEO as the backend."
          config={codexShellCommand}
          language="bash"
        />
      {:else if activeTab === "OpenClaw"}
        {#if instanceAliases.length > 1}
          <div class="text-xs">
            <span
              class="text-xeo-light-gray/50 text-[10px] uppercase tracking-wider block mb-1"
              >Instance</span
            >
            <select bind:value={openClawModel} class={selectClass}>
              {#each instanceAliases as alias}
                <option value={alias}>{alias.split("/").pop()}</option>
              {/each}
            </select>
          </div>
        {/if}
        <IntegrationCard
          title="Config File"
          subtitle="~/.openclaw/openclaw.json"
          description="Add this to your OpenClaw config. If you haven't installed OpenClaw yet, run: npm install -g openclaw@latest"
          config={openClawConfig}
        />
        <IntegrationCard
          title="Setup Commands"
          subtitle="Run in terminal"
          description="After saving the config, run these commands to fix metadata and start the gateway."
          config={`openclaw doctor --fix${capsOf(openClawModel).includes("vision") ? `\nopenclaw models set-image exo/${openClawModel}` : ""}\nopenclaw gateway &\nopenclaw dashboard`}
          language="bash"
        />
      {:else if activeTab === "Pi"}
        {#if instanceAliases.length > 1}
          <div class="text-xs">
            <span
              class="text-xeo-light-gray/50 text-[10px] uppercase tracking-wider block mb-1"
              >Instance</span
            >
            <select bind:value={piModel} class={selectClass}>
              {#each instanceAliases as alias}
                <option value={alias}>{alias.split("/").pop()}</option>
              {/each}
            </select>
          </div>
        {/if}
        <IntegrationCard
          title="Models Config"
          subtitle="~/.pi/agent/models.json"
          description="Register XEO as a custom provider in pi. Create or edit this file, then run pi and pick an XEO model via /model. Install pi with: npm install -g @mariozechner/pi-coding-agent"
          config={piModelsJson}
        />
        <IntegrationCard
          title="Shell Command"
          subtitle="Run in terminal"
          description="Launch pi directly with the XEO provider and model selected."
          config={piShellCommand}
          language="bash"
        />
      {:else if activeTab === "Open WebUI"}
        <IntegrationCard
          title="1. Start Open WebUI"
          subtitle="Run in terminal"
          description="Run this to start Open WebUI."
          config={openWebUiCommand}
          language="bash"
        />
        <IntegrationCard
          title="2. Open & Select Model"
          subtitle="http://localhost:3000"
          description={`Open http://localhost:3000 in your browser. Select the running instance from the dropdown at the top: ${instanceAliases.length > 0 ? instanceAliases.join(", ") : "no instances running"}`}
          config={"open http://localhost:3000"}
          language="bash"
        />
        <IntegrationCard
          title="Ollama CLI"
          subtitle="Run in terminal"
          description="Or use the Ollama CLI directly."
          config={ollamaCommand}
          language="bash"
        />
      {:else if activeTab === "n8n"}
        <IntegrationCard
          title="1. Start n8n"
          subtitle="Run in terminal"
          description="Start n8n with Docker. If you already have n8n running, skip this step."
          config={n8nDockerCommand}
          language="bash"
        />
        <IntegrationCard
          title="2. Open n8n"
          subtitle="http://localhost:5678"
          description="Open n8n in your browser. If this is your first time, complete the setup and select 'Start from Scratch' when prompted."
          config={"open http://localhost:5678"}
          language="bash"
        />
        <IntegrationCard
          title="3. Add OpenAI Credential"
          subtitle="n8n UI → Credentials"
          description="Create an OpenAI credential pointing at your XEO cluster."
          config={n8nCredentialSteps}
        />
        <IntegrationCard
          title="4. Build a Workflow"
          subtitle="n8n UI → Workflows"
          description="Create a workflow that uses your XEO-powered model."
          config={n8nWorkflowSteps}
        />
      {:else if activeTab === "Firefox"}
        <IntegrationCard
          title="Firefox AI Chatbot"
          subtitle="about:config"
          description="Use the XEO dashboard as Firefox's built-in AI chatbot. Requires Firefox 130+."
          config={firefoxConfig}
        />
      {/if}
    </div>
  </main>
</div>
