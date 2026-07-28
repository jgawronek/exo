export interface RebalanceProgress {
  done: number;
  total: number;
}

export interface ActiveRebalance extends RebalanceProgress {
  taskId: string;
  instanceId: string;
}

interface ShiftLayersTask {
  instanceId: string;
  taskStatus: string;
  currentStep: number;
  totalSteps: number;
}

const VISIBLE_REBALANCE_STATUSES = new Set(["Pending", "Running", "Complete"]);

function parseShiftLayersTask(taskWrapped: unknown): ShiftLayersTask | null {
  if (
    typeof taskWrapped !== "object" ||
    taskWrapped === null ||
    !("ShiftLayers" in taskWrapped)
  ) {
    return null;
  }
  const task = taskWrapped.ShiftLayers;
  if (typeof task !== "object" || task === null) return null;

  const instanceId = Reflect.get(task, "instanceId");
  const taskStatus = Reflect.get(task, "taskStatus");
  const currentStep = Reflect.get(task, "currentStep");
  const totalSteps = Reflect.get(task, "totalSteps");
  if (
    typeof instanceId !== "string" ||
    typeof taskStatus !== "string" ||
    typeof currentStep !== "number" ||
    typeof totalSteps !== "number" ||
    currentStep < 1 ||
    totalSteps < currentStep
  ) {
    return null;
  }
  return { instanceId, taskStatus, currentStep, totalSteps };
}

export function getActiveRebalances(
  tasks: Record<string, unknown>,
): Record<string, ActiveRebalance> {
  const rebalances: Record<string, ActiveRebalance> = {};
  for (const [taskId, taskWrapped] of Object.entries(tasks)) {
    const task = parseShiftLayersTask(taskWrapped);
    if (!task || !VISIBLE_REBALANCE_STATUSES.has(task.taskStatus)) continue;

    const done =
      task.taskStatus === "Complete" ? task.currentStep : task.currentStep - 1;
    const candidate: ActiveRebalance = {
      taskId,
      instanceId: task.instanceId,
      done: Math.min(done, task.totalSteps),
      total: task.totalSteps,
    };
    const existing = rebalances[task.instanceId];
    if (!existing || candidate.done > existing.done) {
      rebalances[task.instanceId] = candidate;
    }
  }
  return rebalances;
}

export function formatRebalanceProgress(progress: RebalanceProgress): string {
  return `${progress.done}/${progress.total} layers moved`;
}

/**
 * What a rebalance would actually do, as computed by the backend.
 *
 * The dashboard used to estimate this itself, but its memory-ceiling maths
 * diverged from the allocator's (it ignored the physical-memory budget and
 * treated non-Mac nodes as uncapped), so it advertised speedups the backend
 * would never deliver. The API is now the only source of truth.
 */
export interface RebalancePreview {
  nodeLayers: Record<string, number>;
  steps: number;
  /** Fraction of per-token compute time saved; null when unmeasurable. */
  speedup: number | null;
  currentMs: number | null;
  projectedMs: number | null;
}

/**
 * Ask the backend what a rebalance would produce. Returns null whenever no
 * claim can be made (no timings yet, instance gone, single node) — a failed
 * preview is never surfaced as an error, it just means "say nothing".
 */
export async function fetchRebalancePreview(
  instanceId: string,
  mode: "speed" | "experts" = "speed",
): Promise<RebalancePreview | null> {
  try {
    const response = await fetch(
      `/instance/${instanceId}/rebalance/preview?mode=${mode}`,
    );
    if (!response.ok) return null;
    const body: {
      node_layers?: Record<string, number>;
      steps?: number;
      projected_compute_speedup?: number | null;
      current_compute_ms_per_token?: number | null;
      projected_compute_ms_per_token?: number | null;
    } = await response.json();
    return {
      nodeLayers: body.node_layers ?? {},
      steps: body.steps ?? 0,
      speedup: body.projected_compute_speedup ?? null,
      currentMs: body.current_compute_ms_per_token ?? null,
      projectedMs: body.projected_compute_ms_per_token ?? null,
    };
  } catch {
    return null;
  }
}

export interface RebalanceButtonLabel {
  text: string;
  emphasize: boolean;
  disabled: boolean;
  title: string;
}

const MEANINGFUL_SPEEDUP = 0.01;
const EMPHASIS_SPEEDUP = 0.1;

/**
 * Button copy driven purely by the backend's projection.
 *
 * An unknown projection leaves the button enabled and silent: unknown is not
 * the same as balanced, and pressing it should surface the real error rather
 * than a guess.
 */
export function rebalanceButtonLabel(
  preview: RebalancePreview | null,
  busy: boolean,
): RebalanceButtonLabel {
  if (busy) {
    return {
      text: "REBALANCING...",
      emphasize: false,
      disabled: true,
      title: "Live layer migration in progress",
    };
  }
  if (!preview) {
    return {
      text: "REBALANCE",
      emphasize: false,
      disabled: false,
      title:
        "Live-migrate layers one at a time to the measured-speed split (no downtime)",
    };
  }
  const { speedup, steps, currentMs, projectedMs } = preview;
  if (steps === 0 || speedup === null || speedup < MEANINGFUL_SPEEDUP) {
    return {
      text: "ALREADY BALANCED",
      emphasize: false,
      disabled: true,
      title:
        "The measured-speed split already matches the current one — the fastest node is at its memory ceiling",
    };
  }
  const timing =
    currentMs !== null && projectedMs !== null
      ? `${currentMs.toFixed(0)} → ${projectedMs.toFixed(0)} ms/tok compute, `
      : "";
  return {
    text: `REBALANCE (~${Math.round(speedup * 100)}% FASTER)`,
    emphasize: speedup >= EMPHASIS_SPEEDUP,
    disabled: false,
    // Communication time is unchanged by a re-split, so the end-to-end token
    // rate improves by less than the compute figure.
    title: `${timing}${steps} layer moves. Compute only — token rate gains less.`,
  };
}
