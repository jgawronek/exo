<script lang="ts">
  import { onMount } from "svelte";
  import { fade, fly } from "svelte/transition";
  import { cubicOut } from "svelte/easing";
  import {
    topologyData,
    downloads,
    nodeDisk,
    refreshState,
    lastUpdate as lastUpdateStore,
    startDownload,
    cancelDownload,
    deleteDownload,
    sharedModelsDir,
    sharedModelsDirStatuses,
  } from "$lib/stores/app.svelte";
  import {
    getDownloadTag,
    extractModelIdFromDownload,
    extractShardMetadata,
  } from "$lib/utils/downloads";
  import HeaderNav from "$lib/components/HeaderNav.svelte";

  type CellStatus =
    | { kind: "completed"; totalBytes: number; modelDirectory?: string }
    | {
        kind: "downloading";
        percentage: number;
        downloadedBytes: number;
        totalBytes: number;
        speed: number;
        etaMs: number;
        modelDirectory?: string;
      }
    | {
        kind: "pending";
        downloaded: number;
        total: number;
        modelDirectory?: string;
      }
    | { kind: "failed"; modelDirectory?: string }
    | { kind: "not_present" };

  type ModelCardInfo = {
    family: string;
    quantization: string;
    baseModel: string;
    capabilities: string[];
    storageSize: number;
    nLayers: number;
    supportsTensor: boolean;
  };

  type ModelRow = {
    modelId: string;
    prettyName: string | null;
    cells: Record<string, CellStatus>;
    shardMetadata: Record<string, unknown> | null;
    modelCard: ModelCardInfo | null;
  };

  type NodeColumn = {
    nodeId: string;
    label: string;
    diskAvailable?: number;
    diskTotal?: number;
  };

  const data = $derived(topologyData());
  const downloadsData = $derived(downloads());
  const nodeDiskData = $derived(nodeDisk());

  function getNodeLabel(nodeId: string): string {
    const node = data?.nodes?.[nodeId];
    if (!node) return nodeId.slice(0, 8);
    return (
      node.friendly_name || node.system_info?.model_id || nodeId.slice(0, 8)
    );
  }

  function getBytes(value: unknown): number {
    if (typeof value === "number") return value;
    if (value && typeof value === "object") {
      const v = value as Record<string, unknown>;
      if (typeof v.inBytes === "number") return v.inBytes;
    }
    return 0;
  }

  function formatBytes(bytes: number): string {
    if (!bytes || bytes <= 0) return "0B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.min(
      Math.floor(Math.log(bytes) / Math.log(1024)),
      units.length - 1,
    );
    const val = bytes / Math.pow(1024, i);
    return `${val.toFixed(val >= 10 ? 0 : 1)}${units[i]}`;
  }

  function formatEta(ms: number): string {
    if (!ms || ms <= 0) return "--";
    const totalSeconds = Math.round(ms / 1000);
    const s = totalSeconds % 60;
    const m = Math.floor(totalSeconds / 60) % 60;
    const h = Math.floor(totalSeconds / 3600);
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  }

  function formatSpeed(bytesPerSecond: number): string {
    if (!bytesPerSecond || bytesPerSecond <= 0) return "--";
    const units = ["B/s", "KB/s", "MB/s", "GB/s"];
    const i = Math.min(
      Math.floor(Math.log(bytesPerSecond) / Math.log(1024)),
      units.length - 1,
    );
    const val = bytesPerSecond / Math.pow(1024, i);
    return `${val.toFixed(val >= 10 ? 0 : 1)}${units[i]}`;
  }

  function clampPercent(value: number | undefined): number {
    if (!Number.isFinite(value)) return 0;
    return Math.min(100, Math.max(0, value as number));
  }

  const CELL_PRIORITY: Record<CellStatus["kind"], number> = {
    completed: 4,
    downloading: 3,
    pending: 2,
    failed: 1,
    not_present: 0,
  };

  function shouldUpgradeCell(
    existing: CellStatus,
    candidate: CellStatus,
  ): boolean {
    return CELL_PRIORITY[candidate.kind] > CELL_PRIORITY[existing.kind];
  }

  function extractModelCard(payload: Record<string, unknown>): {
    prettyName: string | null;
    card: ModelCardInfo | null;
  } {
    const shardMetadata = payload.shard_metadata ?? payload.shardMetadata;
    if (!shardMetadata || typeof shardMetadata !== "object")
      return { prettyName: null, card: null };
    const shardObj = shardMetadata as Record<string, unknown>;
    const shardKeys = Object.keys(shardObj);
    if (shardKeys.length !== 1) return { prettyName: null, card: null };
    const shardData = shardObj[shardKeys[0]] as Record<string, unknown>;
    const modelMeta = shardData?.model_card ?? shardData?.modelCard;
    if (!modelMeta || typeof modelMeta !== "object")
      return { prettyName: null, card: null };
    const meta = modelMeta as Record<string, unknown>;

    const prettyName = (meta.prettyName as string) ?? null;

    const card: ModelCardInfo = {
      family: (meta.family as string) ?? "",
      quantization: (meta.quantization as string) ?? "",
      baseModel:
        (meta.base_model as string) ?? (meta.baseModel as string) ?? "",
      capabilities: Array.isArray(meta.capabilities)
        ? (meta.capabilities as string[])
        : [],
      storageSize: getBytes(meta.storage_size ?? meta.storageSize),
      nLayers: (meta.n_layers as number) ?? (meta.nLayers as number) ?? 0,
      supportsTensor:
        (meta.supports_tensor as boolean) ??
        (meta.supportsTensor as boolean) ??
        false,
    };

    return { prettyName, card };
  }

  let modelRows = $state<ModelRow[]>([]);
  let nodeColumns = $state<NodeColumn[]>([]);
  let infoRow = $state<ModelRow | null>(null);

  $effect(() => {
    try {
      if (!downloadsData || Object.keys(downloadsData).length === 0) {
        modelRows = [];
        nodeColumns = [];
        return;
      }

      const allNodeIds = Object.keys(downloadsData);
      const columns: NodeColumn[] = allNodeIds.map((nodeId) => {
        const diskInfo = nodeDiskData?.[nodeId];
        return {
          nodeId,
          label: getNodeLabel(nodeId),
          diskAvailable: diskInfo?.available?.inBytes,
          diskTotal: diskInfo?.total?.inBytes,
        };
      });

      const rowMap = new Map<string, ModelRow>();

      for (const [nodeId, nodeDownloads] of Object.entries(downloadsData)) {
        const entries = Array.isArray(nodeDownloads)
          ? nodeDownloads
          : nodeDownloads && typeof nodeDownloads === "object"
            ? Object.values(nodeDownloads as Record<string, unknown>)
            : [];

        for (const entry of entries) {
          const tagged = getDownloadTag(entry);
          if (!tagged) continue;
          const [tag, payload] = tagged;

          const modelId =
            extractModelIdFromDownload(payload) ?? "unknown-model";
          const { prettyName, card } = extractModelCard(payload);

          if (!rowMap.has(modelId)) {
            rowMap.set(modelId, {
              modelId,
              prettyName,
              cells: {},
              shardMetadata: extractShardMetadata(payload),
              modelCard: card,
            });
          }
          const row = rowMap.get(modelId)!;
          if (prettyName && !row.prettyName) row.prettyName = prettyName;
          if (!row.shardMetadata)
            row.shardMetadata = extractShardMetadata(payload);
          if (!row.modelCard && card) row.modelCard = card;

          const modelDirectory =
            ((payload.model_directory ?? payload.modelDirectory) as string) ||
            undefined;
          let cell: CellStatus;
          if (tag === "DownloadCompleted") {
            const totalBytes = getBytes(payload.total);
            cell = { kind: "completed", totalBytes, modelDirectory };
          } else if (tag === "DownloadOngoing") {
            const rawProgress =
              payload.download_progress ?? payload.downloadProgress ?? {};
            const prog = rawProgress as Record<string, unknown>;
            const totalBytes = getBytes(prog.total ?? payload.total);
            const downloadedBytes = getBytes(prog.downloaded);
            const speed = (prog.speed as number) ?? 0;
            const etaMs =
              (prog.eta_ms as number) ?? (prog.etaMs as number) ?? 0;
            const percentage =
              totalBytes > 0 ? (downloadedBytes / totalBytes) * 100 : 0;
            cell = {
              kind: "downloading",
              percentage: clampPercent(percentage),
              downloadedBytes,
              totalBytes,
              speed,
              etaMs,
              modelDirectory,
            };
          } else if (tag === "DownloadFailed") {
            cell = { kind: "failed", modelDirectory };
          } else {
            const downloaded = getBytes(
              payload.downloaded ??
                payload.downloaded_bytes ??
                payload.downloadedBytes,
            );
            const total = getBytes(
              payload.total ?? payload.total_bytes ?? payload.totalBytes,
            );
            cell = {
              kind: "pending",
              downloaded,
              total,
              modelDirectory,
            };
          }

          const existing = row.cells[nodeId];
          if (!existing || shouldUpgradeCell(existing, cell)) {
            row.cells[nodeId] = cell;
          }
        }
      }

      function rowSortKey(row: ModelRow): number {
        // in progress (4) -> completed (3) -> paused (2) -> not started (1) -> not present (0)
        let best = 0;
        for (const cell of Object.values(row.cells)) {
          let score = 0;
          if (cell.kind === "downloading") score = 4;
          else if (cell.kind === "completed") score = 3;
          else if (cell.kind === "pending" && cell.downloaded > 0)
            score = 2; // paused
          else if (cell.kind === "pending" || cell.kind === "failed") score = 1; // not started
          if (score > best) best = score;
        }
        return best;
      }

      function totalCompletedBytes(row: ModelRow): number {
        let total = 0;
        for (const cell of Object.values(row.cells)) {
          if (cell.kind === "completed") total += cell.totalBytes;
        }
        return total;
      }

      const rows = Array.from(rowMap.values()).sort((a, b) => {
        const aPriority = rowSortKey(a);
        const bPriority = rowSortKey(b);
        if (aPriority !== bPriority) return bPriority - aPriority;
        // Within completed or paused, sort by biggest size first
        if (aPriority === 3 && bPriority === 3) {
          const sizeDiff = totalCompletedBytes(b) - totalCompletedBytes(a);
          if (sizeDiff !== 0) return sizeDiff;
        }
        if (aPriority === 2 && bPriority === 2) {
          const aSize = Math.max(
            ...Object.values(a.cells).map((c) =>
              c.kind === "pending" ? c.total : 0,
            ),
          );
          const bSize = Math.max(
            ...Object.values(b.cells).map((c) =>
              c.kind === "pending" ? c.total : 0,
            ),
          );
          if (aSize !== bSize) return bSize - aSize;
        }
        return a.modelId.localeCompare(b.modelId);
      });

      modelRows = rows;
      nodeColumns = columns;
    } catch (err) {
      console.error("Parse downloads error", err);
      modelRows = [];
      nodeColumns = [];
    }
  });

  const hasDownloads = $derived(modelRows.length > 0);
  const lastUpdateTs = $derived(lastUpdateStore());
  const downloadKeys = $derived(Object.keys(downloadsData || {}));

  // --- Shared model storage ---
  const sharedDir = $derived(sharedModelsDir());
  const sharedDirStatuses = $derived(sharedModelsDirStatuses());
  const clusterNodeIds = $derived(Object.keys(data?.nodes ?? {}));
  const sharedDirValidCount = $derived(
    Object.values(sharedDirStatuses).filter((status) => status.valid).length,
  );
  const sharedDirReportedCount = $derived(
    Object.keys(sharedDirStatuses).length,
  );

  // --- Share mode ---
  // A share is one logical store that each node reaches by its own local path,
  // so nothing here ever compares one node's path against another's.
  type StorageNodeStatus = {
    nodeId: string;
    valid: boolean;
    error?: string | null;
    freeBytes?: number | null;
    path?: string | null;
    mountPath?: string | null;
    writable?: boolean | null;
  };
  type StorageShare = {
    shareId: string;
    mounts: Record<string, string>;
    source?: string | null;
    label?: string | null;
  };

  let storageShare = $state<StorageShare | null>(null);
  let storageMode = $state<"share" | "path" | "none">("none");
  let storageNodes = $state<StorageNodeStatus[]>([]);
  let editingShare = $state(false);
  let shareIdInput = $state("models");
  let shareSourceInput = $state("");
  let shareMountInputs = $state<Record<string, string>>({});
  let shareSaving = $state(false);
  let shareError = $state<string | null>(null);

  async function loadStorageConfig() {
    try {
      const response = await fetch("/models/storage");
      if (!response.ok) return;
      const data = (await response.json()) as {
        share?: StorageShare | null;
        mode?: "share" | "path" | "none";
        perNode?: StorageNodeStatus[];
      };
      storageShare = data.share ?? null;
      storageMode = data.mode ?? "none";
      storageNodes = data.perNode ?? [];
    } catch {
      // Leave the last known config in place; the poll below retries.
    }
  }

  function beginEditShare() {
    shareIdInput = storageShare?.shareId ?? "models";
    shareSourceInput = storageShare?.source ?? "";
    const existing = Object.values(storageShare?.mounts ?? {});
    const commonest = existing
      .slice()
      .sort(
        (a, b) =>
          existing.filter((p) => p === b).length -
          existing.filter((p) => p === a).length,
      )[0];
    shareRootInput = commonest ?? "";
    showShareAdvanced = existing.some((path) => path !== commonest);
    const mounts: Record<string, string> = {};
    for (const nodeId of clusterNodeIds) {
      const configured = storageShare?.mounts?.[nodeId] ?? "";
      mounts[nodeId] = configured === shareRootInput ? "" : configured;
    }
    shareMountInputs = mounts;
    shareError = null;
    editingShare = true;
    loadNetworkServers();
  }

  async function saveShare(clear: boolean) {
    shareSaving = true;
    shareError = null;
    try {
      const body = clear
        ? { shareId: null, mounts: {} }
        : {
            shareId: shareIdInput.trim() || deriveShareId(shareRootInput),
            source:
              shareSourceInput.trim() || deriveShareSource(shareRootInput),
            // A blank override means "same as the shared folder" — or, when a
            // network share was picked instead of a folder, "attach it
            // yourself": nodes auto-mount the source and need no path here.
            mounts: Object.fromEntries(
              clusterNodeIds
                .map((nodeId) => [
                  nodeId,
                  (shareMountInputs[nodeId] || "").trim() ||
                    shareRootInput.trim(),
                ])
                .filter(([, path]) => path),
            ),
          };
      const response = await fetch("/models/storage/share", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        throw new Error(`Request failed (HTTP ${response.status})`);
      }
      editingShare = false;
      await loadStorageConfig();
      await refreshState();
    } catch (err) {
      shareError = err instanceof Error ? err.message : "Failed to save share";
    } finally {
      shareSaving = false;
    }
  }

  let editingSharedDir = $state(false);
  let sharedDirInput = $state("");
  let sharedDirSaving = $state(false);
  let sharedDirError = $state<string | null>(null);
  let browsingSharedDir = $state(false);
  let browseLoading = $state(false);
  let browseError = $state<string | null>(null);
  let browsePath = $state("");
  let browseParentPath = $state<string | null>(null);
  let browseEntries = $state<
    Array<{ name: string; path: string; hidden?: boolean }>
  >([]);
  let browseTruncated = $state(false);
  let browseShowHidden = $state(false);
  let browsePathDraft = $state("");
  let browseNetworkVolumes = $state<
    Array<{
      path: string;
      source: string;
      filesystem: string;
      reachable: boolean;
      label?: string;
    }>
  >([]);
  let browseAvailableShares = $state<
    Array<{ name: string; uri: string; protocol: string; host: string }>
  >([]);
  let mountingUri = $state<string | null>(null);

  async function mountAndBrowse(share: { uri: string }) {
    mountingUri = share.uri;
    browseError = null;
    try {
      const response = await fetch("/models/storage/mount", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uri: share.uri }),
      });
      const data = (await response.json()) as {
        path?: string | null;
        error?: string | null;
      };
      if (data.path) {
        await loadBrowseEntries(data.path);
      } else {
        browseError = data.error ?? "Could not attach the share";
      }
    } catch (err) {
      browseError = err instanceof Error ? err.message : "Mount failed";
    } finally {
      mountingUri = null;
    }
  }

  function beginEditSharedDir() {
    sharedDirInput = sharedDir ?? "";
    sharedDirError = null;
    browsingSharedDir = false;
    editingSharedDir = true;
  }

  function cancelEditSharedDir() {
    editingSharedDir = false;
    browsingSharedDir = false;
    browseError = null;
  }

  async function loadBrowseEntries(path: string | null) {
    browseLoading = true;
    browseError = null;
    try {
      const params = new URLSearchParams();
      if (path && path.length > 0) params.set("path", path);
      if (browseShowHidden) params.set("include_hidden", "true");
      const query = params.toString();
      const response = await fetch(
        `/models/storage/browse${query ? `?${query}` : ""}`,
      );
      if (!response.ok) {
        throw new Error(`Browse failed (HTTP ${response.status})`);
      }
      // The API serialises with a camelCase alias generator, so these keys are
      // camelCase even though the Python fields are snake_case.
      const data = (await response.json()) as {
        path: string;
        parentPath: string | null;
        entries: Array<{ name: string; path: string; hidden?: boolean }>;
        error?: string | null;
        truncated?: boolean;
        networkVolumes?: Array<{
          path: string;
          source: string;
          filesystem: string;
          reachable: boolean;
        }>;
      };
      browsePath = data.path;
      browsePathDraft = data.path;
      browseParentPath = data.parentPath;
      browseEntries = data.entries ?? [];
      browseTruncated = data.truncated ?? false;
      browseNetworkVolumes = data.networkVolumes ?? [];
      browseAvailableShares =
        (
          data as unknown as {
            availableShares?: Array<{
              name: string;
              uri: string;
              protocol: string;
              host: string;
            }>;
          }
        ).availableShares ?? [];
      if (data.error) {
        browseError = data.error;
      }
    } catch (err) {
      browseEntries = [];
      browseTruncated = false;
      browseError =
        err instanceof Error ? err.message : "Failed to browse folders";
    } finally {
      browseLoading = false;
    }
  }

  // Which field the picker writes back to. The picker always browses the node
  // serving the dashboard, so for another node's row it is a way to find a path
  // visually rather than proof that node has it — that node's own status badge
  // is what confirms it.
  let browseTarget = $state<
    | { kind: "legacy" }
    | { kind: "share"; nodeId: string }
    | { kind: "shareRoot" }
  >({ kind: "legacy" });

  let shareRootInput = $state("");
  let showShareAdvanced = $state(false);
  let netServers = $state<Array<{ host: string; name: string }>>([]);
  let netShares = $state<Array<{ name: string; uri: string }>>([]);
  let netSelectedServer = $state<string | null>(null);
  let netServerInput = $state("");
  let netLoading = $state(false);
  let netError = $state<string | null>(null);

  async function loadNetworkServers() {
    netLoading = true;
    netError = null;
    try {
      const response = await fetch("/models/storage/network");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = (await response.json()) as {
        servers?: Array<{ host: string; name: string }>;
      };
      netServers = data.servers ?? [];
    } catch (err) {
      netError = err instanceof Error ? err.message : "Discovery failed";
    } finally {
      netLoading = false;
    }
  }

  async function loadNetworkShares(host: string) {
    netLoading = true;
    netError = null;
    netSelectedServer = host;
    netShares = [];
    try {
      const response = await fetch(
        `/models/storage/network?server=${encodeURIComponent(host)}`,
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = (await response.json()) as {
        shares?: Array<{ name: string; uri: string }>;
        error?: string | null;
      };
      netShares = data.shares ?? [];
      if (data.error) netError = data.error;
    } catch (err) {
      netError = err instanceof Error ? err.message : "Could not list shares";
    } finally {
      netLoading = false;
    }
  }

  function pickNetworkShare(share: { name: string; uri: string }) {
    // The nodes attach the share themselves; no paths to fill in.
    shareIdInput = share.name;
    shareSourceInput = share.uri;
    shareRootInput = "";
    shareMountInputs = Object.fromEntries(
      clusterNodeIds.map((nodeId) => [nodeId, ""]),
    );
  }

  async function openShareRootBrowser() {
    browseTarget = { kind: "shareRoot" };
    browsingSharedDir = true;
    await loadBrowseEntries(shareRootInput.trim() || null);
  }

  // A folder on a network mount already knows where it came from, so fill the
  // source in rather than making the user retype what the mount table says.
  function deriveShareSource(path: string): string | null {
    const volume = browseNetworkVolumes
      .filter((v) => path === v.path || path.startsWith(`${v.path}/`))
      .sort((a, b) => b.path.length - a.path.length)[0];
    if (!volume) return null;
    const scheme = /smb|cifs/i.test(volume.filesystem) ? "smb" : "nfs";
    return `${scheme}://${volume.source}`;
  }

  function deriveShareId(path: string): string {
    const name = path.replace(/\/+$/, "").split("/").pop();
    return name || "models";
  }

  function applyShareRoot(path: string) {
    shareRootInput = path;
    shareSourceInput = deriveShareSource(path) ?? "";
  }

  async function openSharedDirBrowser() {
    browseTarget = { kind: "legacy" };
    browsingSharedDir = true;
    await loadBrowseEntries(sharedDirInput.trim() || null);
  }

  async function openShareMountBrowser(nodeId: string) {
    browseTarget = { kind: "share", nodeId };
    browsingSharedDir = true;
    await loadBrowseEntries(shareMountInputs[nodeId]?.trim() || null);
  }

  async function goToTypedPath() {
    await loadBrowseEntries(browsePathDraft.trim() || null);
  }

  async function toggleBrowseHidden() {
    browseShowHidden = !browseShowHidden;
    await loadBrowseEntries(browsePath || null);
  }

  // Whatever the path bar shows wins, so a typed path can be used without
  // navigating to it first.
  const chosenBrowsePath = $derived(browsePathDraft.trim() || browsePath);

  function closeSharedDirBrowser() {
    browsingSharedDir = false;
    browseError = null;
  }

  function selectBrowseFolder(path: string) {
    if (browseTarget.kind === "shareRoot") {
      applyShareRoot(path);
      closeSharedDirBrowser();
      return;
    }
    if (browseTarget.kind === "share") {
      shareMountInputs = { ...shareMountInputs, [browseTarget.nodeId]: path };
    } else {
      sharedDirInput = path;
    }
    closeSharedDirBrowser();
  }

  function handleBrowseKeydown(event: KeyboardEvent) {
    if (!browsingSharedDir) return;
    if (event.key === "Escape") {
      event.preventDefault();
      closeSharedDirBrowser();
    }
  }

  async function saveSharedDir(path: string | null) {
    sharedDirSaving = true;
    sharedDirError = null;
    try {
      const response = await fetch("/models/storage", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      if (!response.ok) {
        throw new Error(`Request failed (HTTP ${response.status})`);
      }
      editingSharedDir = false;
      browsingSharedDir = false;
      await refreshState();
    } catch (err) {
      sharedDirError =
        err instanceof Error ? err.message : "Failed to update location";
    } finally {
      sharedDirSaving = false;
    }
  }

  onMount(() => {
    refreshState();
    loadStorageConfig();
    // Nodes validate their own path asynchronously, so keep polling for their
    // verdicts rather than showing a snapshot taken before they replied.
    const storagePoll = setInterval(loadStorageConfig, 5000);
    return () => clearInterval(storagePoll);
  });
</script>

<svelte:window onkeydown={handleBrowseKeydown} />

{#snippet trashIcon()}
  <svg
    class="w-5 h-5"
    viewBox="0 0 20 20"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
  >
    <path
      d="M4 6h12M8 6V4h4v2m1 0v10a1 1 0 01-1 1H8a1 1 0 01-1-1V6h6"
      stroke-linecap="round"
      stroke-linejoin="round"
    ></path>
  </svg>
{/snippet}

{#snippet downloadIcon(size?: string)}
  <svg
    class={size ?? "w-5 h-5"}
    viewBox="0 0 20 20"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
  >
    <path
      d="M10 3v10m0 0l-3-3m3 3l3-3M3 17h14"
      stroke-linecap="round"
      stroke-linejoin="round"
    ></path>
  </svg>
{/snippet}

{#snippet pauseIcon()}
  <svg class="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
    <path
      fill-rule="evenodd"
      d="M6 4h2v12H6V4zm6 0h2v12h-2V4z"
      clip-rule="evenodd"
    ></path>
  </svg>
{/snippet}

{#snippet deleteButton(nodeId: string, modelId: string)}
  <button
    type="button"
    class="text-white/50 hover:text-red-400 transition-colors cursor-pointer"
    onclick={() => deleteDownload(nodeId, modelId)}
    title="Delete from this node"
  >
    {@render trashIcon()}
  </button>
{/snippet}

<div class="min-h-screen bg-xeo-dark-gray text-white">
  <HeaderNav showHome={true} />
  <div class="max-w-7xl mx-auto px-4 lg:px-8 py-6 space-y-6">
    <div class="flex items-center justify-between gap-4 flex-wrap">
      <div>
        <h1
          class="text-2xl font-mono tracking-[0.2em] uppercase text-xeo-green"
        >
          Library
        </h1>
        <p class="text-sm text-xeo-light-gray">
          Models available on your nodes and downloads in progress
        </p>
      </div>
      <div class="flex items-center gap-3">
        <button
          type="button"
          class="text-xs font-mono text-xeo-light-gray hover:text-xeo-green transition-colors uppercase border border-xeo-medium-gray/40 px-2 py-1 rounded"
          onclick={() => refreshState()}
          title="Force refresh from /state"
        >
          Refresh
        </button>
        <div class="text-[11px] font-mono text-xeo-light-gray">
          Last update: {lastUpdateTs
            ? new Date(lastUpdateTs).toLocaleTimeString()
            : "n/a"}
        </div>
      </div>
    </div>

    <!-- Shared model storage -->
    <div
      class="rounded border border-xeo-medium-gray/30 bg-xeo-black/30 p-4 space-y-3"
    >
      <div class="flex items-center justify-between gap-3 flex-wrap">
        <div
          class="text-xs font-mono tracking-[0.2em] uppercase text-xeo-green flex items-center gap-2"
        >
          <span class="w-1.5 h-1.5 bg-xeo-green rounded-full"></span>
          Shared Model Storage
          {#if storageMode === "share"}
            <span
              class="rounded-sm border border-cyan-400/30 bg-cyan-400/10 px-1 py-0.5 text-[9px] tracking-wider text-cyan-300"
              >SHARE</span
            >
          {/if}
        </div>
        {#if sharedDir && sharedDirReportedCount > 0}
          <div
            class="text-[11px] font-mono {sharedDirValidCount ===
            sharedDirReportedCount
              ? 'text-green-400'
              : 'text-yellow-400'}"
          >
            valid on {sharedDirValidCount}/{Math.max(
              sharedDirReportedCount,
              clusterNodeIds.length,
            )} nodes
          </div>
        {/if}
      </div>

      <!-- Share mode: one share, a different local path per node -->
      {#if editingShare}
        <div
          class="rounded border border-cyan-400/25 bg-cyan-400/5 p-3 space-y-3"
        >
          <div class="space-y-1.5">
            <div
              class="text-[10px] font-mono uppercase tracking-wider text-cyan-300/70"
            >
              On the network
            </div>
            <div class="flex items-center gap-2 flex-wrap">
              {#each netServers as server (server.host)}
                <button
                  type="button"
                  disabled={netLoading}
                  onclick={() => loadNetworkShares(server.host)}
                  class="text-xs font-mono px-2.5 py-1 rounded border transition-colors disabled:opacity-50 cursor-pointer {netSelectedServer ===
                  server.host
                    ? 'border-cyan-400/60 text-cyan-300 bg-cyan-400/10'
                    : 'border-xeo-medium-gray/50 text-white/70 hover:text-white'}"
                >
                  {server.name}
                  <span class="text-white/35">{server.host}</span>
                </button>
              {/each}
              <input
                type="text"
                bind:value={netServerInput}
                placeholder="or a server address"
                class="w-44 bg-xeo-black/60 border border-xeo-medium-gray/50 rounded px-2 py-1 text-xs font-mono text-white placeholder:text-white/25 focus:outline-none focus:border-cyan-400/60"
              />
              <button
                type="button"
                disabled={netLoading || !netServerInput.trim()}
                onclick={() => loadNetworkShares(netServerInput.trim())}
                class="text-[10px] font-mono uppercase tracking-wider px-2 py-1 rounded border border-cyan-400/40 text-cyan-300 hover:bg-cyan-400/10 transition-colors disabled:opacity-50 cursor-pointer"
              >
                Connect
              </button>
            </div>
            {#if netLoading}
              <div class="text-[11px] font-mono text-white/40">
                Looking around&hellip;
              </div>
            {:else if netSelectedServer && netShares.length > 0}
              <div class="flex items-center gap-2 flex-wrap">
                {#each netShares as share (share.uri)}
                  <button
                    type="button"
                    onclick={() => pickNetworkShare(share)}
                    class="text-xs font-mono px-2.5 py-1 rounded border transition-colors cursor-pointer {shareSourceInput ===
                    share.uri
                      ? 'border-cyan-400 text-xeo-black bg-cyan-400'
                      : 'border-cyan-400/40 text-cyan-300 hover:bg-cyan-400/10'}"
                  >
                    {share.name}
                  </button>
                {/each}
              </div>
            {:else if netSelectedServer}
              <div class="text-[11px] font-mono text-white/40">
                No guest shares on {netSelectedServer}
              </div>
            {/if}
            {#if netError}
              <div class="text-[11px] font-mono text-red-400">{netError}</div>
            {/if}
            {#if shareSourceInput.startsWith("smb://")}
              <div class="text-[11px] font-mono text-cyan-300/80">
                {shareSourceInput} — every node attaches this automatically.
              </div>
            {/if}
          </div>

          <div class="flex items-center gap-2 flex-wrap">
            <label
              class="text-[10px] font-mono uppercase tracking-wider text-white/50"
              for="share-root-input">Or a folder</label
            >
            <input
              id="share-root-input"
              type="text"
              value={shareRootInput}
              oninput={(event) =>
                applyShareRoot((event.currentTarget as HTMLInputElement).value)}
              placeholder="pick a folder every node can read"
              class="flex-1 min-w-[240px] bg-xeo-black/60 border border-cyan-400/30 rounded px-2 py-1.5 text-xs font-mono text-white placeholder:text-white/25 focus:outline-none focus:border-cyan-400/60"
            />
            <button
              type="button"
              disabled={browseLoading}
              onclick={openShareRootBrowser}
              class="flex-shrink-0 text-xs font-mono uppercase tracking-wider px-3 py-1.5 rounded bg-cyan-400 text-xeo-black hover:bg-cyan-300 transition-colors disabled:opacity-50 cursor-pointer"
            >
              Browse
            </button>
          </div>

          {#if shareError}
            <div class="text-[11px] font-mono text-red-400">{shareError}</div>
          {/if}

          <div class="flex items-center gap-2 flex-wrap">
            <button
              type="button"
              disabled={shareSaving ||
                (!shareRootInput.trim() && !shareSourceInput.trim())}
              onclick={() => saveShare(false)}
              class="text-xs font-mono uppercase tracking-wider px-3 py-1.5 rounded bg-cyan-400 text-xeo-black hover:bg-cyan-300 transition-colors disabled:opacity-50 cursor-pointer"
            >
              {shareSaving ? "Saving…" : "Save"}
            </button>
            <button
              type="button"
              disabled={shareSaving}
              onclick={() => (editingShare = false)}
              class="text-xs font-mono uppercase tracking-wider px-3 py-1.5 rounded border border-xeo-medium-gray/50 text-xeo-light-gray hover:text-white transition-colors cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="button"
              onclick={() => (showShareAdvanced = !showShareAdvanced)}
              class="text-[10px] font-mono uppercase tracking-wider text-white/35 hover:text-white/60 transition-colors cursor-pointer"
            >
              {showShareAdvanced ? "Hide" : "Per-node paths"}
            </button>
          </div>

          <!-- Only for the uncommon case: a node that reaches the same share
               at a different path. Hidden by default so the simple flow is
               browse, select, save. -->
          {#if showShareAdvanced}
            <div class="space-y-1.5 pt-2 border-t border-xeo-medium-gray/30">
              <div class="text-[10px] font-mono text-white/35">
                Every node uses the folder above unless overridden here.
              </div>
              {#each clusterNodeIds as nodeId (nodeId)}
                <div class="flex items-center gap-2">
                  <span
                    class="w-40 flex-shrink-0 truncate text-xs font-mono text-white/70"
                    title={nodeId}>{getNodeLabel(nodeId)}</span
                  >
                  <input
                    type="text"
                    bind:value={shareMountInputs[nodeId]}
                    placeholder="same as above"
                    class="flex-1 min-w-0 bg-xeo-black/60 border border-xeo-medium-gray/50 rounded px-2 py-1 text-xs font-mono text-white placeholder:text-white/25 focus:outline-none focus:border-cyan-400/60"
                  />
                  <button
                    type="button"
                    disabled={browseLoading}
                    onclick={() => openShareMountBrowser(nodeId)}
                    class="flex-shrink-0 text-[10px] font-mono uppercase tracking-wider px-2 py-1 rounded border border-cyan-400/40 text-cyan-300 hover:bg-cyan-400/10 transition-colors disabled:opacity-50 cursor-pointer"
                  >
                    Browse
                  </button>
                </div>
              {/each}
            </div>
          {/if}
        </div>
      {:else if storageMode === "share" && storageShare}
        <div
          class="rounded border border-cyan-400/25 bg-cyan-400/5 p-3 space-y-2"
        >
          <div class="flex items-center justify-between gap-3 flex-wrap">
            <div class="text-xs font-mono text-white/85">
              {storageShare.shareId}
              {#if storageShare.source}
                <span class="text-white/40"> · {storageShare.source}</span>
              {/if}
            </div>
            <div class="flex items-center gap-2">
              <button
                type="button"
                onclick={beginEditShare}
                class="text-[10px] font-mono uppercase tracking-wider px-2 py-1 rounded border border-cyan-400/40 text-cyan-300 hover:bg-cyan-400/10 transition-colors cursor-pointer"
                >Edit</button
              >
              <button
                type="button"
                disabled={shareSaving}
                onclick={() => saveShare(true)}
                class="text-[10px] font-mono uppercase tracking-wider px-2 py-1 rounded border border-xeo-medium-gray/50 text-xeo-light-gray hover:text-white transition-colors disabled:opacity-50 cursor-pointer"
                >Clear</button
              >
            </div>
          </div>
          <div class="text-[10px] font-mono text-white/40">
            Models here are available on top of each node's own downloads.
            Choose one and every node loads it from the share instead of
            downloading its own copy.
          </div>
          <div class="space-y-1">
            {#each storageNodes as node (node.nodeId)}
              <div class="flex items-center gap-2 text-[11px] font-mono">
                <span
                  class="w-1.5 h-1.5 rounded-full flex-shrink-0 {node.valid
                    ? 'bg-green-400'
                    : 'bg-red-400'}"
                ></span>
                <span class="w-40 flex-shrink-0 truncate text-white/70"
                  >{getNodeLabel(node.nodeId)}</span
                >
                <span class="flex-1 min-w-0 truncate text-white/50"
                  >{node.path ?? node.mountPath ?? "—"}</span
                >
                {#if node.valid && node.writable === false}
                  <span
                    class="flex-shrink-0 rounded-sm border border-cyan-400/30 bg-cyan-400/10 px-1 py-0.5 text-[9px] tracking-wider text-cyan-300"
                    >READ-ONLY</span
                  >
                {/if}
                <span
                  class="flex-shrink-0 {node.valid
                    ? 'text-green-400'
                    : 'text-red-400'}"
                >
                  {node.valid
                    ? node.freeBytes != null
                      ? `${formatBytes(node.freeBytes)} free`
                      : "ready"
                    : (node.error ?? "unavailable")}
                </span>
              </div>
            {/each}
          </div>
        </div>
      {:else}
        <button
          type="button"
          onclick={beginEditShare}
          class="text-[10px] font-mono uppercase tracking-wider px-2 py-1 rounded border border-cyan-400/40 text-cyan-300 hover:bg-cyan-400/10 transition-colors cursor-pointer"
        >
          Add shared storage
        </button>
      {/if}

      {#if editingSharedDir}
        <form
          class="flex items-center gap-2 flex-wrap"
          onsubmit={(event) => {
            event.preventDefault();
            saveSharedDir(sharedDirInput.trim() || null);
          }}
        >
          <!-- svelte-ignore a11y_autofocus -->
          <input
            type="text"
            bind:value={sharedDirInput}
            placeholder="/mnt/models (same mount path on every node)"
            autofocus
            class="flex-1 min-w-[260px] bg-xeo-black/60 border border-xeo-medium-gray/50 rounded px-3 py-1.5 text-xs font-mono text-white placeholder:text-white/30 focus:outline-none focus:border-xeo-green/60"
          />
          <button
            type="button"
            disabled={sharedDirSaving || browseLoading}
            onclick={openSharedDirBrowser}
            class="text-xs font-mono uppercase tracking-wider px-3 py-1.5 rounded border border-xeo-green/40 text-xeo-green hover:bg-xeo-green/10 transition-colors disabled:opacity-50 cursor-pointer"
          >
            Browse
          </button>
          <button
            type="submit"
            disabled={sharedDirSaving}
            class="text-xs font-mono uppercase tracking-wider px-3 py-1.5 rounded bg-xeo-green text-xeo-black hover:bg-xeo-green-darker transition-colors disabled:opacity-50 cursor-pointer"
          >
            {sharedDirSaving ? "Saving..." : "Save"}
          </button>
          <button
            type="button"
            disabled={sharedDirSaving}
            onclick={cancelEditSharedDir}
            class="text-xs font-mono uppercase tracking-wider px-3 py-1.5 rounded border border-xeo-medium-gray/50 text-xeo-light-gray hover:text-white transition-colors cursor-pointer"
          >
            Cancel
          </button>
        </form>
      {:else}
        <div class="flex items-center gap-3 flex-wrap">
          {#if sharedDir}
            <span class="text-sm font-mono text-white">{sharedDir}</span>
          {:else}
            <span class="text-sm font-mono text-white/40"
              >Not set &mdash; each node uses its local directory</span
            >
          {/if}
          <button
            type="button"
            onclick={beginEditSharedDir}
            class="text-xs font-mono uppercase tracking-wider px-2 py-1 rounded border border-xeo-medium-gray/40 text-xeo-light-gray hover:text-xeo-green hover:border-xeo-green/50 transition-colors cursor-pointer"
          >
            Edit
          </button>
          {#if sharedDir}
            <button
              type="button"
              disabled={sharedDirSaving}
              onclick={() => saveSharedDir(null)}
              class="text-xs font-mono uppercase tracking-wider px-2 py-1 rounded border border-red-500/30 text-red-400/80 hover:text-red-400 hover:border-red-500/50 transition-colors disabled:opacity-50 cursor-pointer"
            >
              Clear
            </button>
          {/if}
        </div>
      {/if}

      {#if sharedDirError}
        <div class="text-[11px] font-mono text-red-400">{sharedDirError}</div>
      {/if}

      {#if sharedDir}
        {#if sharedDirReportedCount === 0}
          <div class="text-[11px] font-mono text-white/50">
            Waiting for nodes to verify the location&hellip;
          </div>
        {:else}
          <div class="flex flex-wrap gap-2">
            {#each Object.entries(sharedDirStatuses) as [nodeId, status] (nodeId)}
              <div
                class="flex items-center gap-1.5 px-2 py-1 rounded border text-[11px] font-mono {status.valid
                  ? 'border-green-500/30 text-green-400 bg-green-500/5'
                  : 'border-red-500/30 text-red-400 bg-red-500/5'}"
                title={status.valid
                  ? `Valid${status.freeBytes != null ? ` · ${formatBytes(status.freeBytes)} free` : ""}`
                  : (status.error ?? "Invalid")}
              >
                <span
                  class="w-1.5 h-1.5 rounded-full {status.valid
                    ? 'bg-green-400'
                    : 'bg-red-400'}"
                ></span>
                <span class="text-white/80">{getNodeLabel(nodeId)}</span>
                {#if status.valid}
                  <span>
                    VALID{status.freeBytes != null
                      ? ` · ${formatBytes(status.freeBytes)} free`
                      : ""}
                  </span>
                {:else}
                  <span>INVALID{status.error ? ` · ${status.error}` : ""}</span>
                {/if}
              </div>
            {/each}
            {#each clusterNodeIds.filter((nodeId) => !(nodeId in sharedDirStatuses)) as nodeId (nodeId)}
              <div
                class="flex items-center gap-1.5 px-2 py-1 rounded border border-xeo-medium-gray/40 text-[11px] font-mono text-white/50"
              >
                <span class="w-1.5 h-1.5 rounded-full bg-white/30"></span>
                <span>{getNodeLabel(nodeId)}</span>
                <span>CHECKING&hellip;</span>
              </div>
            {/each}
          </div>
        {/if}
        <div class="text-[10px] font-mono text-white/40">
          Models load from and download to this location when it is valid on a
          node; otherwise that node falls back to its local directory. The path
          must be the same mount point on every node.
        </div>
      {/if}
    </div>

    {#if !hasDownloads}
      <div
        class="rounded border border-xeo-medium-gray/30 bg-xeo-black/30 p-6 text-center text-xeo-light-gray space-y-2"
      >
        <div class="text-sm">
          No models found. Start a model download to add it to your library.
        </div>
        <div class="text-[11px] text-xeo-light-gray/70">
          Download keys detected: {downloadKeys.length === 0
            ? "none"
            : downloadKeys.join(", ")}
        </div>
      </div>
    {:else}
      <div
        class="rounded border border-xeo-medium-gray/30 bg-xeo-black/30 overflow-x-auto"
      >
        <table class="w-full text-left font-mono text-xs">
          <thead>
            <tr class="border-b border-xeo-medium-gray/30">
              <th
                class="sticky left-0 z-10 bg-xeo-black px-4 py-3 text-[11px] uppercase tracking-wider text-xeo-green font-medium whitespace-nowrap border-r border-xeo-medium-gray/20"
              >
                Model
              </th>
              {#each nodeColumns as col}
                <th
                  class="px-4 py-3 text-[11px] uppercase tracking-wider text-xeo-light-gray font-medium text-center whitespace-nowrap min-w-[120px]"
                >
                  <div>{col.label}</div>
                  {#if col.diskAvailable != null}
                    <div
                      class="text-[9px] text-white/70 normal-case tracking-normal mt-0.5"
                    >
                      {formatBytes(col.diskAvailable)} free
                    </div>
                  {/if}
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each modelRows as row}
              <tr
                class="group border-b border-xeo-medium-gray/20 hover:bg-xeo-medium-gray/10 transition-colors"
              >
                <td
                  class="sticky left-0 z-10 bg-xeo-dark-gray group-hover:bg-[oklch(0.18_0_0)] transition-colors px-4 py-3 whitespace-nowrap border-r border-xeo-medium-gray/20"
                >
                  <div class="flex items-center gap-2">
                    <div class="min-w-0">
                      <div class="text-white text-xs" title={row.modelId}>
                        {row.prettyName ?? row.modelId}
                      </div>
                      {#if row.prettyName}
                        <div
                          class="text-[10px] text-white/60"
                          title={row.modelId}
                        >
                          {row.modelId}
                        </div>
                      {/if}
                    </div>
                    <button
                      type="button"
                      class="p-1 rounded hover:bg-white/10 transition-colors flex-shrink-0 opacity-60 group-hover:opacity-100"
                      onclick={() => (infoRow = row)}
                      title="View model details"
                    >
                      <svg
                        class="w-4 h-4 text-white/60 hover:text-white/80"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                      >
                        <path
                          d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"
                        />
                      </svg>
                    </button>
                  </div>
                </td>

                {#each nodeColumns as col}
                  {@const cell = row.cells[col.nodeId] ?? {
                    kind: "not_present" as const,
                  }}
                  <td class="px-4 py-3 text-center align-middle">
                    {#if cell.kind === "completed"}
                      <div
                        class="flex flex-col items-center gap-1"
                        title="Completed ({formatBytes(cell.totalBytes)})"
                      >
                        <svg
                          class="w-7 h-7 text-green-400"
                          viewBox="0 0 20 20"
                          fill="currentColor"
                        >
                          <path
                            fill-rule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clip-rule="evenodd"
                          ></path>
                        </svg>
                        <span class="text-xs text-white/70"
                          >{formatBytes(cell.totalBytes)}</span
                        >
                        {@render deleteButton(col.nodeId, row.modelId)}
                      </div>
                    {:else if cell.kind === "downloading"}
                      <div
                        class="flex flex-col items-center gap-1"
                        title="{formatBytes(
                          cell.downloadedBytes,
                        )} / {formatBytes(cell.totalBytes)} - {formatSpeed(
                          cell.speed,
                        )} - ETA {formatEta(cell.etaMs)}"
                      >
                        <span class="text-xeo-green text-sm font-medium"
                          >{clampPercent(cell.percentage).toFixed(1)}%</span
                        >
                        <div
                          class="w-16 h-2 bg-xeo-black/60 rounded-sm overflow-hidden"
                        >
                          <div
                            class="h-full bg-gradient-to-r from-xeo-green to-xeo-green/70 transition-all duration-300"
                            style="width: {clampPercent(
                              cell.percentage,
                            ).toFixed(1)}%"
                          ></div>
                        </div>
                        <span class="text-[10px] text-white/70"
                          >{formatSpeed(cell.speed)}</span
                        >
                        <div class="flex gap-1 mt-0.5">
                          <button
                            type="button"
                            class="text-white/50 hover:text-xeo-green transition-colors cursor-pointer"
                            onclick={() =>
                              cancelDownload(col.nodeId, row.modelId)}
                            title="Pause download"
                          >
                            {@render pauseIcon()}
                          </button>
                          {@render deleteButton(col.nodeId, row.modelId)}
                        </div>
                      </div>
                    {:else if cell.kind === "pending"}
                      <div
                        class="flex flex-col items-center gap-1"
                        title={cell.downloaded > 0
                          ? `${formatBytes(cell.downloaded)} / ${formatBytes(cell.total)} downloaded (paused)`
                          : "Download pending"}
                      >
                        {#if cell.downloaded > 0 && cell.total > 0}
                          <span class="text-white/70 text-xs"
                            >{formatBytes(cell.downloaded)} / {formatBytes(
                              cell.total,
                            )}</span
                          >
                          <div
                            class="w-full h-1.5 bg-white/10 rounded-full overflow-hidden"
                          >
                            <div
                              class="h-full bg-xeo-light-gray/40 rounded-full"
                              style="width: {(
                                (cell.downloaded / cell.total) *
                                100
                              ).toFixed(1)}%"
                            ></div>
                          </div>
                          <div class="flex gap-1">
                            {#if row.shardMetadata}
                              <button
                                type="button"
                                class="text-white/50 hover:text-xeo-green transition-colors cursor-pointer"
                                onclick={() =>
                                  startDownload(col.nodeId, row.shardMetadata!)}
                                title="Resume download on this node"
                              >
                                {@render downloadIcon()}
                              </button>
                            {:else}
                              <span class="text-white/50 text-[10px]"
                                >paused</span
                              >
                            {/if}
                            {@render deleteButton(col.nodeId, row.modelId)}
                          </div>
                        {:else if row.shardMetadata}
                          <button
                            type="button"
                            class="text-white/50 hover:text-xeo-green transition-colors cursor-pointer"
                            onclick={() =>
                              startDownload(col.nodeId, row.shardMetadata!)}
                            title="Start download on this node"
                          >
                            {@render downloadIcon("w-6 h-6")}
                          </button>
                        {:else}
                          <span class="text-white/40 text-sm">...</span>
                        {/if}
                      </div>
                    {:else if cell.kind === "failed"}
                      <div
                        class="flex flex-col items-center gap-1"
                        title="Download failed"
                      >
                        <svg
                          class="w-7 h-7 text-red-400"
                          viewBox="0 0 20 20"
                          fill="currentColor"
                        >
                          <path
                            fill-rule="evenodd"
                            d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                            clip-rule="evenodd"
                          ></path>
                        </svg>
                        <div class="flex gap-1">
                          {#if row.shardMetadata}
                            <button
                              type="button"
                              class="text-white/50 hover:text-xeo-green transition-colors cursor-pointer"
                              onclick={() =>
                                startDownload(col.nodeId, row.shardMetadata!)}
                              title="Retry download on this node"
                            >
                              {@render downloadIcon()}
                            </button>
                          {/if}
                          {@render deleteButton(col.nodeId, row.modelId)}
                        </div>
                      </div>
                    {:else}
                      <div
                        class="flex flex-col items-center"
                        title="Not on this node"
                      >
                        <span class="text-xeo-medium-gray text-lg leading-none"
                          >--</span
                        >
                        {#if row.shardMetadata}
                          <button
                            type="button"
                            class="text-white/50 hover:text-xeo-green transition-colors mt-0.5 opacity-0 group-hover:opacity-100 cursor-pointer"
                            onclick={() =>
                              startDownload(col.nodeId, row.shardMetadata!)}
                            title="Download to this node"
                          >
                            {@render downloadIcon()}
                          </button>
                        {/if}
                      </div>
                    {/if}
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </div>
</div>

<!-- Shared models folder browse modal -->
{#if browsingSharedDir}
  <div
    class="fixed inset-0 z-[60] bg-black/80 backdrop-blur-sm"
    transition:fade={{ duration: 150 }}
    onclick={closeSharedDirBrowser}
    role="presentation"
  ></div>
  <div
    class="fixed z-[60] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[min(92vw,560px)] h-[min(80vh,640px)] bg-xeo-dark-gray border border-xeo-green/10 rounded-lg shadow-2xl overflow-hidden flex flex-col"
    transition:fly={{ y: 16, duration: 220, easing: cubicOut }}
    role="dialog"
    aria-modal="true"
    aria-label="Browse shared model folder"
  >
    <div
      class="flex items-start justify-between gap-3 p-4 border-b border-xeo-green/10 bg-xeo-medium-gray/30"
    >
      <div class="min-w-0">
        <h3
          class="text-sm font-mono tracking-[0.15em] uppercase text-xeo-green"
        >
          Browse Folder
        </h3>
        <p class="mt-1 text-[11px] font-mono text-white/45">
          Folders and network volumes on the node serving this page.
        </p>
      </div>
      <button
        type="button"
        class="p-1 rounded hover:bg-white/10 transition-colors text-white/50 cursor-pointer"
        onclick={closeSharedDirBrowser}
        title="Close"
        aria-label="Close browse dialog"
      >
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
          <path
            d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"
          />
        </svg>
      </button>
    </div>

    <div class="px-4 py-3 border-b border-xeo-medium-gray/30 space-y-2">
      <form
        class="flex items-center gap-2"
        onsubmit={(event) => {
          event.preventDefault();
          goToTypedPath();
        }}
      >
        {#if browsePath !== ""}
          <button
            type="button"
            disabled={browseLoading}
            onclick={() => loadBrowseEntries(browseParentPath)}
            class="text-[10px] font-mono uppercase tracking-wider px-2 py-1 rounded border border-xeo-medium-gray/40 text-xeo-light-gray hover:text-white transition-colors disabled:opacity-50 cursor-pointer flex-shrink-0"
          >
            Up
          </button>
        {/if}
        <input
          type="text"
          bind:value={browsePathDraft}
          spellcheck="false"
          autocomplete="off"
          placeholder="Type or paste any path, e.g. /mnt/models"
          aria-label="Path to browse"
          class="flex-1 min-w-0 px-3 py-1.5 rounded bg-xeo-black/50 border border-xeo-medium-gray/40 text-xs font-mono text-white/85 placeholder:text-white/25 focus:outline-none focus:border-xeo-green/60"
        />
        <button
          type="submit"
          disabled={browseLoading}
          class="text-[10px] font-mono uppercase tracking-wider px-2 py-1 rounded border border-xeo-green/40 text-xeo-green hover:bg-xeo-green/10 transition-colors disabled:opacity-50 cursor-pointer flex-shrink-0"
        >
          Go
        </button>
      </form>
      <div class="flex items-center justify-between gap-3">
        <div class="text-[10px] font-mono text-white/35 truncate">
          {browsePath || "Shortcuts — open / to reach any folder"}
        </div>
        <button
          type="button"
          disabled={browseLoading}
          onclick={toggleBrowseHidden}
          class="text-[10px] font-mono uppercase tracking-wider px-2 py-1 rounded border transition-colors disabled:opacity-50 cursor-pointer flex-shrink-0 {browseShowHidden
            ? 'border-xeo-green/50 text-xeo-green bg-xeo-green/10'
            : 'border-xeo-medium-gray/40 text-xeo-light-gray hover:text-white'}"
        >
          {browseShowHidden ? "Hiding nothing" : "Show hidden"}
        </button>
      </div>
    </div>

    <div class="flex-1 overflow-y-auto">
      {#if !browseLoading && browseNetworkVolumes.length > 0}
        <div class="border-b border-xeo-medium-gray/30 bg-xeo-black/30">
          <div
            class="px-4 pt-3 pb-1 text-[10px] font-mono uppercase tracking-wider text-cyan-300/70"
          >
            LAN shares this node has mounted
          </div>
          {#each browseNetworkVolumes as volume (volume.path)}
            <button
              type="button"
              disabled={!volume.reachable}
              onclick={() => loadBrowseEntries(volume.path)}
              class="w-full flex items-start gap-3 px-4 py-2 text-left transition-colors border-t border-xeo-medium-gray/15 {volume.reachable
                ? 'hover:bg-cyan-400/10 cursor-pointer'
                : 'opacity-60 cursor-not-allowed'}"
            >
              <svg
                class="w-4 h-4 mt-0.5 flex-shrink-0 {volume.reachable
                  ? 'text-cyan-300/80'
                  : 'text-red-400/70'}"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.75"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M4 5h16v6H4zM4 15h16v4H4zM8 8h.01M8 17h.01"
                />
              </svg>
              <span class="min-w-0 flex-1">
                <span class="flex items-center gap-2">
                  <span class="text-sm font-mono text-white/85 truncate"
                    >{volume.label || volume.path}</span
                  >
                  <span
                    class="flex-shrink-0 rounded-sm border border-cyan-400/30 bg-cyan-400/10 px-1 py-0.5 text-[9px] font-mono tracking-wider text-cyan-300"
                    >{volume.filesystem.toUpperCase()}</span
                  >
                  {#if !volume.reachable}
                    <span
                      class="flex-shrink-0 rounded-sm border border-red-400/30 bg-red-400/10 px-1 py-0.5 text-[9px] font-mono tracking-wider text-red-300"
                      >UNREACHABLE</span
                    >
                  {/if}
                </span>
                <span
                  class="block text-[10px] font-mono text-white/40 truncate"
                  title={`${volume.source} mounted at ${volume.path}`}
                  >{volume.source}</span
                >
              </span>
            </button>
          {/each}
        </div>
      {/if}
      {#if !browseLoading && browsePath === "" && browseAvailableShares.length > 0}
        <div class="border-b border-xeo-medium-gray/30 bg-xeo-black/20">
          <div
            class="px-4 pt-3 pb-1 text-[10px] font-mono uppercase tracking-wider text-cyan-300/60"
          >
            Available on the LAN — not mounted here yet
          </div>
          {#each browseAvailableShares as share (share.uri)}
            <div
              class="w-full flex items-center gap-3 px-4 py-2 border-t border-xeo-medium-gray/15"
            >
              <svg
                class="w-4 h-4 flex-shrink-0 text-white/35"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.75"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M4 5h16v6H4zM4 15h16v4H4zM8 8h.01M8 17h.01"
                />
              </svg>
              <span class="min-w-0 flex-1">
                <span class="flex items-center gap-2">
                  <span class="text-sm font-mono text-white/70 truncate"
                    >{share.name} · {share.host}</span
                  >
                  <span
                    class="flex-shrink-0 rounded-sm border border-xeo-medium-gray/40 bg-white/5 px-1 py-0.5 text-[9px] font-mono tracking-wider text-white/50"
                    >{share.protocol.toUpperCase()}</span
                  >
                </span>
                <span class="block text-[10px] font-mono text-white/30 truncate"
                  >{share.uri}</span
                >
              </span>
              {#if share.protocol === "smb"}
                <button
                  type="button"
                  disabled={mountingUri !== null}
                  onclick={() => mountAndBrowse(share)}
                  class="flex-shrink-0 text-[10px] font-mono uppercase tracking-wider px-2 py-1 rounded border border-cyan-400/40 text-cyan-300 hover:bg-cyan-400/10 transition-colors disabled:opacity-50 cursor-pointer"
                >
                  {mountingUri === share.uri ? "Mounting…" : "Mount & browse"}
                </button>
              {:else}
                <span
                  class="flex-shrink-0 text-[10px] font-mono uppercase tracking-wider text-white/30"
                  title="NFS needs a one-time root mount on this node; exo cannot do it unprivileged"
                  >root mount needed</span
                >
              {/if}
            </div>
          {/each}
        </div>
      {/if}
      {#if !browseLoading && browsePath === "" && browseNetworkVolumes.length > 0}
        <div
          class="px-4 pt-3 pb-1 text-[10px] font-mono uppercase tracking-wider text-white/40"
        >
          Local folders on this node
        </div>
      {/if}
      {#if browseLoading}
        <div
          class="h-full flex items-center justify-center text-xs font-mono text-white/50"
        >
          Loading folders&hellip;
        </div>
      {:else if browseError && browseEntries.length === 0}
        <div
          class="h-full flex items-center justify-center px-6 text-center text-xs font-mono text-red-400"
        >
          {browseError}
        </div>
      {:else if browseEntries.length === 0}
        <div
          class="h-full flex items-center justify-center text-xs font-mono text-white/40"
        >
          No subfolders here
        </div>
      {:else}
        {#each browseEntries as entry (entry.path)}
          <button
            type="button"
            onclick={() => loadBrowseEntries(entry.path)}
            class="w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm font-mono text-white/85 hover:bg-xeo-green/10 hover:text-xeo-green transition-colors border-b border-xeo-medium-gray/15 last:border-b-0 cursor-pointer"
          >
            <svg
              class="w-4 h-4 text-xeo-green/70 flex-shrink-0"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.75"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V7z"
              />
            </svg>
            <span class="truncate {entry.hidden ? 'text-white/50' : ''}"
              >{entry.name}</span
            >
          </button>
        {/each}
      {/if}
    </div>

    {#if browseError && browseEntries.length > 0}
      <div
        class="px-4 py-2 text-[11px] font-mono text-yellow-400 border-t border-xeo-medium-gray/30"
      >
        {browseError}
      </div>
    {/if}

    {#if browseTruncated}
      <div
        class="px-4 py-2 text-[11px] font-mono text-yellow-400 border-t border-xeo-medium-gray/30"
      >
        Showing the first {browseEntries.length} folders — type a path above to jump
        further in.
      </div>
    {/if}

    <div
      class="flex items-center justify-between gap-3 p-4 border-t border-xeo-green/10 bg-xeo-medium-gray/20"
    >
      <p class="text-[10px] font-mono text-white/40 max-w-[55%]">
        A network folder has to be mounted on a node before that node can read
        it.
      </p>
      <div class="flex items-center gap-2 flex-shrink-0">
        <button
          type="button"
          onclick={closeSharedDirBrowser}
          class="text-xs font-mono uppercase tracking-wider px-3 py-1.5 rounded border border-xeo-medium-gray/50 text-xeo-light-gray hover:text-white transition-colors cursor-pointer"
        >
          Cancel
        </button>
        <button
          type="button"
          disabled={browseLoading || !chosenBrowsePath}
          onclick={() => selectBrowseFolder(chosenBrowsePath)}
          class="text-xs font-mono uppercase tracking-wider px-3 py-1.5 rounded bg-xeo-green text-xeo-black hover:bg-xeo-green-darker transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
        >
          Use this folder
        </button>
      </div>
    </div>
  </div>
{/if}

<!-- Info modal -->
{#if infoRow}
  <div
    class="fixed inset-0 z-[60] bg-black/60"
    transition:fade={{ duration: 150 }}
    onclick={() => (infoRow = null)}
    role="presentation"
  ></div>
  <div
    class="fixed z-[60] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[min(80vw,400px)] bg-xeo-dark-gray border border-xeo-green/10 rounded-lg shadow-2xl p-4"
    transition:fly={{ y: 10, duration: 200, easing: cubicOut }}
    role="dialog"
    aria-modal="true"
  >
    <div class="flex items-start justify-between mb-3">
      <h3 class="font-mono text-lg text-white">
        {infoRow.prettyName ?? infoRow.modelId}
      </h3>
      <button
        type="button"
        class="p-1 rounded hover:bg-white/10 transition-colors text-white/50"
        onclick={() => (infoRow = null)}
        title="Close model details"
        aria-label="Close info dialog"
      >
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
          <path
            d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"
          />
        </svg>
      </button>
    </div>
    <div class="space-y-2 text-xs font-mono">
      <div class="flex items-center gap-2">
        <span class="text-white/40">Model ID:</span>
        <span class="text-white/70">{infoRow.modelId}</span>
      </div>
      {#if infoRow.modelCard}
        {#if infoRow.modelCard.family}
          <div class="flex items-center gap-2">
            <span class="text-white/40">Family:</span>
            <span class="text-white/70">{infoRow.modelCard.family}</span>
          </div>
        {/if}
        {#if infoRow.modelCard.baseModel}
          <div class="flex items-center gap-2">
            <span class="text-white/40">Base model:</span>
            <span class="text-white/70">{infoRow.modelCard.baseModel}</span>
          </div>
        {/if}
        {#if infoRow.modelCard.quantization}
          <div class="flex items-center gap-2">
            <span class="text-white/40">Quantization:</span>
            <span class="text-white/70">{infoRow.modelCard.quantization}</span>
          </div>
        {/if}
        {#if infoRow.modelCard.storageSize > 0}
          <div class="flex items-center gap-2">
            <span class="text-white/40">Size:</span>
            <span class="text-white/70"
              >{formatBytes(infoRow.modelCard.storageSize)}</span
            >
          </div>
        {/if}
        {#if infoRow.modelCard.nLayers > 0}
          <div class="flex items-center gap-2">
            <span class="text-white/40">Layers:</span>
            <span class="text-white/70">{infoRow.modelCard.nLayers}</span>
          </div>
        {/if}
        {#if infoRow.modelCard.capabilities.length > 0}
          <div class="flex items-center gap-2">
            <span class="text-white/40">Capabilities:</span>
            <span class="text-white/70"
              >{infoRow.modelCard.capabilities.join(", ")}</span
            >
          </div>
        {/if}
        <div class="flex items-center gap-2">
          <span class="text-white/40">Tensor parallelism:</span>
          <span class="text-white/70"
            >{infoRow.modelCard.supportsTensor ? "Yes" : "No"}</span
          >
        </div>
      {/if}

      <!-- Per-node download status -->
      {#if nodeColumns.filter((col) => (infoRow?.cells[col.nodeId]?.kind ?? "not_present") !== "not_present").length > 0}
        <div class="mt-3 pt-3 border-t border-xeo-green/10">
          <div class="flex items-center gap-2 mb-1">
            <svg
              class="w-3.5 h-3.5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path
                class="text-white/40"
                d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"
              />
              <path class="text-green-400" d="m9 13 2 2 4-4" />
            </svg>
            <span class="text-white/40">On nodes:</span>
          </div>
          <div class="flex flex-col gap-1.5 mt-1">
            {#each nodeColumns as col}
              {@const cellStatus = infoRow?.cells[col.nodeId]}
              {#if cellStatus && cellStatus.kind !== "not_present"}
                <div class="flex flex-col gap-0.5">
                  <span
                    class="inline-block w-fit px-1.5 py-0.5 rounded text-[10px] {cellStatus.kind ===
                    'completed'
                      ? 'bg-green-500/10 text-green-400/80 border border-green-500/20'
                      : cellStatus.kind === 'downloading'
                        ? 'bg-xeo-green/10 text-xeo-green/80 border border-xeo-green/20'
                        : cellStatus.kind === 'failed'
                          ? 'bg-red-500/10 text-red-400/80 border border-red-500/20'
                          : 'bg-white/5 text-white/50 border border-white/10'}"
                  >
                    {col.label}
                    {#if cellStatus.kind === "downloading" && "percentage" in cellStatus}
                      ({clampPercent(cellStatus.percentage).toFixed(0)}%)
                    {/if}
                  </span>
                  {#if "modelDirectory" in cellStatus && cellStatus.modelDirectory}
                    <span
                      class="text-[9px] text-white/30 break-all pl-1"
                      title={cellStatus.modelDirectory}
                    >
                      {cellStatus.modelDirectory}
                    </span>
                  {/if}
                </div>
              {/if}
            {/each}
          </div>
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  table {
    min-width: max-content;
  }
</style>
