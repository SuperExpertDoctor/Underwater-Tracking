import { useMemo } from "react";
import type { RegionalPlanView } from "../../types/frames";
import "./RegionTaskGraph.css";

export const REGION_NODE_WIDTH = 96;
export const REGION_NODE_HEIGHT = 44;
export const ENTITY_NODE_RADIUS = 14;

export type RegionGraphNode = {
  id: string;
  kind: "region" | "entity";
  shape: "square" | "circle";
  label: string;
  x: number;
  y: number;
  status?: string;
  platform?: "uuv";
};

export type RegionGraphEdge = {
  id: string;
  kind: "temporal" | "responsibility";
  source: string;
  target: string;
};

export type RegionGraphLayout = {
  width: number;
  height: number;
  nodes: RegionGraphNode[];
  edges: RegionGraphEdge[];
};

const GRAPH_PADDING = 36;
const REGION_GAP = 30;

function entityLabel(id: string): string {
  const match = id.match(/^uuv[_-]?0*(\d+)$/i);
  return match ? `UUV_${Number(match[1])}` : id;
}

function regionLabel(displayName: string, ordinal: number): string {
  const match = displayName.match(/(?:region|区域)[_\s-]?(\d+)$/i);
  const value = match ? Number(match[1]) : ordinal + 1;
  return `R${String(value).padStart(2, "0")}`;
}

export function buildRegionGraphLayout(
  plan: RegionalPlanView | null | undefined,
  minimumWidth = 720,
): RegionGraphLayout {
  if (!plan || plan.regions.length === 0) return { width: minimumWidth, height: 260, nodes: [], edges: [] };

  const regions = [...plan.regions].sort(
    (left, right) => left.start_time_s - right.start_time_s || left.region_id.localeCompare(right.region_id),
  );
  const regionStep = REGION_NODE_WIDTH + REGION_GAP;
  const width = Math.max(minimumWidth, GRAPH_PADDING * 2 + regions.length * regionStep - REGION_GAP);
  const regionY = 66;
  const nodes: RegionGraphNode[] = regions.map((region, index) => ({
    id: `region:${region.region_id}`,
    kind: "region",
    shape: "square",
    label: regionLabel(region.display_name, index),
    x: GRAPH_PADDING + REGION_NODE_WIDTH / 2 + index * regionStep,
    y: regionY,
    status: region.effect.status,
  }));
  const regionNodeById = new Map(regions.map((region, index) => [region.region_id, nodes[index]]));

  const entityIds = new Set<string>();
  regions.forEach((region) => {
    region.assigned_uuv_ids.forEach((id) => entityIds.add(id));
  });
  const entities = [...entityIds].sort();
  const entityX = (index: number, total: number) => total === 1
    ? width / 2
    : GRAPH_PADDING + ENTITY_NODE_RADIUS + index * ((width - GRAPH_PADDING * 2 - ENTITY_NODE_RADIUS * 2) / (total - 1));
  const entityNodes = entities.map((id) => {
    return {
      id: `entity:${id}`,
      kind: "entity" as const,
      shape: "circle" as const,
      label: entityLabel(id),
      x: entityX(entities.indexOf(id), entities.length),
      y: 146,
      platform: "uuv" as const,
    };
  });
  nodes.push(...entityNodes);

  const edges: RegionGraphEdge[] = [];
  regions.forEach((region) => {
    region.successor_region_ids.forEach((successorId) => {
      if (!regionNodeById.has(successorId)) return;
      edges.push({
        id: `temporal:${region.region_id}:${successorId}`,
        kind: "temporal",
        source: `region:${region.region_id}`,
        target: `region:${successorId}`,
      });
    });
    region.assigned_uuv_ids.forEach((entityId) => {
      edges.push({
        id: `responsibility:${entityId}:${region.region_id}`,
        kind: "responsibility",
        source: `entity:${entityId}`,
        target: `region:${region.region_id}`,
      });
    });
  });
  return { width, height: 258, nodes, edges };
}

interface RegionTaskGraphProps {
  plan: RegionalPlanView | null | undefined;
  selectedRegionId?: string | null;
  selectedEntityId?: string | null;
  onSelectRegion?: (regionId: string | null) => void;
  onSelectEntity?: (entityId: string | null) => void;
}

export default function RegionTaskGraph({
  plan,
  selectedRegionId = null,
  selectedEntityId = null,
  onSelectRegion,
  onSelectEntity,
}: RegionTaskGraphProps) {
  const layout = useMemo(() => buildRegionGraphLayout(plan), [plan]);
  const nodeById = useMemo(() => new Map(layout.nodes.map((node) => [node.id, node])), [layout.nodes]);
  const selectNode = (node: RegionGraphNode) => {
    if (node.kind === "region") {
      const regionId = node.id.slice("region:".length);
      onSelectRegion?.(selectedRegionId === regionId ? null : regionId);
    }
    else onSelectEntity?.(node.id.slice("entity:".length));
  };

  if (!plan || layout.nodes.length === 0) return <div className="region-graph-empty" role="status">等待目标预测区域和编组任务。</div>;

  return <div className="region-graph-scroll">
    <svg
      className="region-task-graph"
      role="img"
      aria-label={`${plan.target_id} 区域接力知识图谱`}
      viewBox={`0 0 ${layout.width} ${layout.height}`}
      style={{ width: layout.width }}
    >
      <defs>
        <marker id="region-graph-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="currentColor" />
        </marker>
      </defs>
      {layout.edges.map((edge) => {
        const source = nodeById.get(edge.source);
        const target = nodeById.get(edge.target);
        if (!source || !target) return null;
        return <line
          key={edge.id}
          className={`region-graph-edge ${edge.kind}`}
          data-edge-kind={edge.kind}
          x1={source.x}
          y1={source.y}
          x2={target.x}
          y2={target.y}
          markerEnd="url(#region-graph-arrow)"
          aria-label={edge.kind === "temporal" ? "区域时间接力" : "UUV 跟踪责任"}
        />;
      })}
      {layout.nodes.map((node) => {
        const selected = node.kind === "region"
          ? selectedRegionId === node.id.slice("region:".length)
          : selectedEntityId === node.id.slice("entity:".length);
        return <g
          key={node.id}
          className={`region-graph-node ${node.kind} ${node.shape} status-${node.status ?? ""} ${selected ? "selected" : ""}`}
          data-node-shape={node.shape}
          data-node-id={node.id}
          role="button"
          tabIndex={0}
          aria-label={node.kind === "region" ? `区域 ${node.label}` : `实体 ${node.label}`}
          aria-pressed={selected}
          onClick={() => selectNode(node)}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              selectNode(node);
            }
          }}
        >
          {node.shape === "square"
            ? <rect x={node.x - REGION_NODE_WIDTH / 2} y={node.y - REGION_NODE_HEIGHT / 2} width={REGION_NODE_WIDTH} height={REGION_NODE_HEIGHT} rx="4" />
            : <circle cx={node.x} cy={node.y} r={ENTITY_NODE_RADIUS} />}
          <text x={node.x} y={node.y + 3} textAnchor="middle">{node.label}</text>
          <title>{node.kind === "region" ? node.label : `${node.platform?.toUpperCase()} ${node.label}`}</title>
        </g>;
      })}
    </svg>
    <div className="region-graph-legend" aria-label="图谱图例">
      <span><i className="legend-square" />预测区域</span>
      <span><i className="legend-circle legend-uuv" />UUV</span>
      <span><i className="legend-line temporal" />时间接力</span>
      <span><i className="legend-line active-tracking" />UUV 跟踪责任</span>
    </div>
  </div>;
}
