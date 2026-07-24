/**
 * Derive the placement-chosen MLX ring route from instance shard order.
 *
 * Pipeline deviceRank order matches place_instance + order_cycle_for_fastest_links.
 */

export type RingHop = {
  source: string;
  target: string;
};

type ShardAssignments = {
  nodeToRunner?: Record<string, string>;
  runnerToShard?: Record<string, unknown>;
  modelId?: string;
};

type RingInstance = {
  shardAssignments?: ShardAssignments;
};

function unwrapTagged(wrapped: unknown): [string | null, unknown] {
  if (!wrapped || typeof wrapped !== "object") return [null, null];
  const keys = Object.keys(wrapped as Record<string, unknown>);
  if (keys.length === 1) {
    return [keys[0], (wrapped as Record<string, unknown>)[keys[0]]];
  }
  return [null, null];
}

/**
 * Return node IDs in pipeline ring order (deviceRank ascending).
 */
export function getPipelineRingNodeIds(instance: unknown): string[] {
  if (!instance || typeof instance !== "object") return [];
  const assignments = (instance as RingInstance).shardAssignments;
  if (!assignments) return [];

  const runnerToShard = assignments.runnerToShard || {};
  const nodeToRunner = assignments.nodeToRunner || {};

  const runnerEntries = Object.entries(runnerToShard).map(
    ([runnerId, shardWrapped]) => {
      const [tag, shard] = unwrapTagged(shardWrapped);
      const meta = shard as
        | { modelMeta?: { deviceRank?: number } }
        | undefined;
      return {
        runnerId,
        tag,
        deviceRank: meta?.modelMeta?.deviceRank ?? 0,
      };
    },
  );

  return runnerEntries
    .filter((entry) => entry.tag === "PipelineShardMetadata")
    .sort((a, b) => a.deviceRank - b.deviceRank)
    .map((entry) => {
      const nodeId = Object.entries(nodeToRunner).find(
        ([, runnerId]) => runnerId === entry.runnerId,
      )?.[0];
      return nodeId;
    })
    .filter((nodeId): nodeId is string => Boolean(nodeId));
}

/** Directed ring hops: rank i → rank (i+1) % N. */
export function getRingRouteHops(ringNodeIds: string[]): RingHop[] {
  if (ringNodeIds.length < 2) return [];
  return ringNodeIds.map((source, index) => ({
    source,
    target: ringNodeIds[(index + 1) % ringNodeIds.length],
  }));
}

function getInstanceModelId(instance: unknown): string | null {
  if (!instance || typeof instance !== "object") return null;
  return (instance as RingInstance).shardAssignments?.modelId ?? null;
}

type PlacementPreviewLike = {
  model_id: string;
  sharding: "Pipeline" | "Tensor";
  instance_meta: "MlxRing" | "MlxJaccl";
  instance: unknown | null;
};

/**
 * Prefer a running MlxRingInstance (matching selected model when possible),
 * else the selected placement preview's ring instance.
 */
export function resolvePipelineRingNodeIds(
  instances: Record<string, unknown> | null | undefined,
  placementPreviews: PlacementPreviewLike[] | null | undefined,
  selectedPreviewModelId: string | null | undefined,
): string[] {
  const runningRings: Array<{ modelId: string | null; nodeIds: string[] }> = [];

  for (const instanceWrapped of Object.values(instances || {})) {
    const [tag, instance] = unwrapTagged(instanceWrapped);
    if (tag !== "MlxRingInstance") continue;
    const nodeIds = getPipelineRingNodeIds(instance);
    if (nodeIds.length >= 2) {
      runningRings.push({
        modelId: getInstanceModelId(instance),
        nodeIds,
      });
    }
  }

  if (runningRings.length > 0) {
    if (selectedPreviewModelId) {
      const matching = runningRings.find(
        (ring) => ring.modelId === selectedPreviewModelId,
      );
      if (matching) return matching.nodeIds;
    }
    return runningRings[0].nodeIds;
  }

  if (!selectedPreviewModelId || !placementPreviews) return [];

  const preview = placementPreviews.find(
    (entry) =>
      entry.model_id === selectedPreviewModelId &&
      entry.instance_meta === "MlxRing" &&
      entry.sharding === "Pipeline" &&
      entry.instance != null,
  );
  if (!preview?.instance) return [];

  const [tag, instance] = unwrapTagged(preview.instance);
  if (tag && tag !== "MlxRingInstance") {
    // Preview may store the bare instance or a tagged wrapper
    const bareIds = getPipelineRingNodeIds(preview.instance);
    if (bareIds.length >= 2) return bareIds;
  }
  const nodeIds = getPipelineRingNodeIds(instance ?? preview.instance);
  return nodeIds.length >= 2 ? nodeIds : [];
}

/** Put ring nodes first (in route order), then any remaining cluster nodes. */
export function orderNodeIdsByRing(
  allNodeIds: string[],
  ringNodeIds: string[],
): string[] {
  if (ringNodeIds.length === 0) return allNodeIds;
  const allSet = new Set(allNodeIds);
  const ringSet = new Set(ringNodeIds);
  const inRing = ringNodeIds.filter((id) => allSet.has(id));
  const rest = allNodeIds.filter((id) => !ringSet.has(id));
  return [...inRing, ...rest];
}
