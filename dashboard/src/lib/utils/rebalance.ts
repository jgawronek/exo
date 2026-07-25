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
