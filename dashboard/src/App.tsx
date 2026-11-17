import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import { toast, Toaster } from "sonner";
import {
  Activity, ArrowDownRight, ArrowUpRight, Brain, CircleDot, Gauge, Moon,
  ShieldAlert, ShieldCheck, Sun, Terminal, Timer, Waves,
} from "lucide-react";
import * as Drawer from "vaul";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { BorderTrail } from "@/components/ui/border-trail";
import { TextShimmer } from "@/components/ui/text-shimmer";
import {
  MorphingDialog, MorphingDialogClose, MorphingDialogContainer, MorphingDialogContent,
  MorphingDialogDescription, MorphingDialogTitle, MorphingDialogTrigger,
} from "@/components/ui/morphing-dialog";
import { Expandable, ExpandableContent, ExpandableTrigger } from "@/components/ui/expandable";
import { SparkAreaChart, SparkLineChart } from "@/components/SparkChart/SparkChart";
import { ProgressCircle } from "@/components/ProgressCircle/ProgressCircle";
import { Tracker } from "@/components/Tracker/Tracker";
import { Pill, PillIndicator } from "@/components/kibo-ui/pill";
import { Island } from "@/chrome/Island";
import { Figure, Tile, Unit } from "@/chrome/Instrument";
import { ThemeProvider, useTheme } from "@/chrome/theme";
import { glide, layout, reveal } from "@/motion";
import { useMediaQuery } from "@/lib/useMediaQuery";
import { cn } from "@/lib/utils";

/* ---------------------------------------------------------------- types */

type Position = { side: string; entry: string; contracts: number; tp: string; sl: string; opened_ts: number; mark: string | null };
type Order = { side: string; price: string; contracts: number; age_s: number; filled?: string };
type RestingExit = { price: string; contracts: number; filled: string; reason: string };
type Reconcile = { generated: string; since: number; venue_net: string; ledger_net: string; delta: string; ts: number };
type Trade = {
  id: number; series: string; side: string; entry: string; exit_price: string; contracts: number;
  fee: string; pnl: string; exit_reason: string; closed_ts: number; p_model: number;
};
type Slot = {
  id: string; series: string; asset: string; display: string; state: string; reason: string;
  halted: boolean; ticker: string | null; strike: number | null; spot: number; spot_age_s: number;
  p_up: number; sigma: number; yes_bid: string | null; yes_ask: string | null; last: string | null;
  spread: string | null; secs_left: number; window_open_ts: number; window_close_ts: number;
  edge: string; confidence: number; bankroll: string; pnl_today: string; wins: number; losses: number;
  streak: number; ece: number; params: Record<string, string>; position: Position | null;
  order: Order | null; last_trade: { pnl: string; reason: string; ts: number } | null;
  gate: { available: boolean; allow: boolean; confidence: number; note: string; latency_ms: number; error: string | null } | null;
  signal?: string;
  signal_detail?: string;
  exit?: RestingExit | null;
  window_entries?: number;
  max_window_trades?: number;
  tape: { ts: number; p: number; mid: number; edge: number; spot: number }[];
};
type Program = {
  id: number; ts: number; slot: string; name: string; status: string; score: number;
  profit_factor: number; ece: number; note: string; generation: number;
};
type Snapshot = {
  generated_at: number; mode: string; env: string; live_armed: boolean; uptime_s: number;
  health: { coinbase: boolean; kraken: boolean; spot_messages: number; market_errors: number; latency_ms: number; venue_calls: number; last_error: string | null; feed_errors: string[] };
  account: { equity: string; start_equity: string; cash: string; realized: string; day_pnl: string; open_positions: number; peak_equity: string };
  curve: { ts: number; equity: number }[];
  slots: Slot[];
  fills: { ts: number; series: string; side: string; kind: string; price: string; contracts: number; pnl: string; reason: string; fee?: string | number }[];
  capacity?: { open: number; max: number };
  epoch_start?: number;
  reconcile?: Reconcile | null;
  trades: Trade[];
  shr: {
    state: string; generation: number; promoted: number; rejected: number;
    fault: { reason: string; ts: number; ticks: number } | null;
    candidates: { name: string; status: string; score: number; profit_factor: number; ece: number; note: string }[];
    timeline: { ts: number; state: string; detail: string }[];
    programs: Program[];
  };
  window: { open_ts: number; close_ts: number };
  venue_balance: string; armed_at: number | null; kill_reason: string | null;
  readiness: Readiness;
  gate: { url: string; calls: number; failures: number; mode?: string; last: { available: boolean; allow: boolean; confidence: number; note: string; model: string; latency_ms: number; error: string | null } | null };
  ws?: { messages: number; lag_ms: number; degraded: boolean; books: number; indices: number; reconnects: number; sync_problems: number; final_minute_index: number };
};

type Check = { key: string; label: string; ok: boolean; detail: string };
type Readiness = {
  ready: boolean; armed: boolean; armed_at: number | null; kill_reason: string | null;
  elapsed_minutes: number; settled_trades: number; net: string;
  blockers: string[]; checks: Check[];
  strategy?: Record<string, unknown> | null;
  controls?: { ok: boolean; detail: string } | null;
  ws?: { messages: number; lag_ms: number; degraded: boolean; gaps: number; unrecovered_gaps: number } | null;
  canary?: { ok: boolean; reconciled: boolean; flattened: boolean; ticker: string } | null;
  evaluated_ms?: number;
};

/* ---------------------------------------------------------------- normalize */

/* The snapshot is the contract; a screen that assumes a field exists goes blank the
   first time the API changes shape. Every value the render dereferences is given a
   defined default at the fetch boundary, so a renamed or dropped field reads as a
   dash instead of unmounting the tree. */
const finite = (v: unknown, fallback = 0) => (Number.isFinite(Number(v)) ? Number(v) : fallback);

function normalizeSnapshot(raw: unknown): Snapshot {
  const next = (raw ?? {}) as Partial<Snapshot>;
  const r = (next.readiness ?? {}) as Partial<Readiness>;
  const checks: Check[] = Array.isArray(r.checks)
    ? r.checks
        .filter((c) => Boolean(c) && typeof c === "object")
        .map((c) => ({
          key: String(c.key ?? ""),
          label: String(c.label ?? c.key ?? "check"),
          ok: Boolean(c.ok),
          detail: String(c.detail ?? ""),
        }))
    : [];
  return {
    ...(next as Snapshot),
    readiness: {
      ready: Boolean(r.ready),
      armed: Boolean(r.armed),
      armed_at: r.armed_at ?? null,
      kill_reason: r.kill_reason ?? null,
      elapsed_minutes: finite(r.elapsed_minutes),
      settled_trades: finite(r.settled_trades),
      net: String(r.net ?? "0"),
      blockers: Array.isArray(r.blockers) ? r.blockers.map(String) : [],
      checks,
      strategy: r.strategy ?? null,
      controls: r.controls ?? null,
      ws: r.ws ?? null,
      canary: r.canary ?? null,
      evaluated_ms: finite(r.evaluated_ms),
    },
    slots: Array.isArray(next.slots) ? next.slots : [],
    fills: Array.isArray(next.fills) ? next.fills : [],
    trades: Array.isArray(next.trades) ? next.trades : [],
    curve: Array.isArray(next.curve) ? next.curve : [],
    gate: next.gate ?? { url: "", calls: 0, failures: 0, last: null },
    shr: {
      state: String(next.shr?.state ?? "RUNNING"),
      generation: finite(next.shr?.generation),
      promoted: finite(next.shr?.promoted),
      rejected: finite(next.shr?.rejected),
      fault: next.shr?.fault ?? null,
      candidates: Array.isArray(next.shr?.candidates) ? next.shr.candidates : [],
      timeline: Array.isArray(next.shr?.timeline) ? next.shr.timeline : [],
      programs: Array.isArray(next.shr?.programs) ? next.shr.programs : [],
    },
  };
}

/* ---------------------------------------------------------------- format */

const usd = (v: string | number, digits = 2) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return `${n < 0 ? "−" : ""}$${Math.abs(n).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
};
const cents = (v: string | number) => `${(Number(v) * 100).toFixed(1)}¢`;
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const clock = (ts: number) => new Date(ts * 1000).toLocaleTimeString("en-GB", { hour12: false, timeZone: "UTC" });
const mmss = (s: number) => `${Math.floor(Math.max(0, s) / 60)}:${String(Math.floor(Math.max(0, s) % 60)).padStart(2, "0")}`;
/* A fill's liquidity side is read off its fee: the venue charges makers nothing. */
const feeSide = (fee: string | number | undefined) =>
  fee === undefined ? null : Number(fee) > 0 ? { label: "taker", fee: Number(fee) } : { label: "maker", fee: 0 };
const pctOf = (part: number, whole: number) => (whole <= 0 ? 0 : Math.max(0, Math.min(100, (part / whole) * 100)));

const STATE_TONE: Record<string, { label: string; tone: "inactive" | "info" | "warn" | "active" }> = {
  abstain: { label: "abstain", tone: "inactive" },
  wait: { label: "waiting", tone: "inactive" },
  hold: { label: "working", tone: "active" },
  momentum_burst: { label: "momentum", tone: "info" },
  mean_revert: { label: "mean revert", tone: "info" },
  exit: { label: "exiting", tone: "active" },
  halted: { label: "halted", tone: "warn" },
  "no market": { label: "no contract", tone: "inactive" },
  boot: { label: "booting", tone: "inactive" },
  quoting: { label: "quoting", tone: "info" },
};
const TONE_CLASS = {
  inactive: "bg-pill-inactive-bg text-pill-inactive-fg",
  info: "bg-tag-ops-bg text-tag-ops-fg",
  warn: "bg-pill-warn-bg text-pill-warn-fg",
  active: "bg-pill-active-bg text-pill-active-fg",
};

function slotPhase(s: Slot) {
  if (s.halted) return "halted";
  if (s.position) return "hold";
  if (s.order) return "quoting";
  return s.state;
}

/* ---------------------------------------------------------------- window rail */

function WindowRail({ slot, now }: { slot: Slot; now: number }) {
  const open = slot.window_open_ts || slot.window_close_ts - 900;
  const close = slot.window_close_ts || open + 900;
  const total = Math.max(1, close - open);
  const nowPct = pctOf(now - open, total);
  const startPct = slot.position ? pctOf(slot.position.opened_ts - open, total)
    : slot.order ? pctOf(now - slot.order.age_s - open, total) : null;

  return (
    <div className="relative h-9 w-full overflow-hidden rounded-chip bg-track">
      <div className="absolute inset-y-0 left-0 bg-blue-soft" style={{ width: `${nowPct}%` }} />
      {[0.25, 0.5, 0.75].map((t) => (
        <span key={t} aria-hidden className="absolute inset-y-1 w-px bg-hairline" style={{ left: `${t * 100}%` }} />
      ))}
      {startPct !== null && (
        <div
          className={cn("absolute inset-y-2 rounded-full", slot.position ? "bg-blue" : "bg-blue/25")}
          style={{ left: `${startPct}%`, width: `${Math.max(0.8, nowPct - startPct)}%` }}
        />
      )}
      <motion.span
        aria-hidden
        layout
        transition={layout}
        className="absolute inset-y-0 w-[2px] bg-ink"
        style={{ left: `${nowPct}%` }}
      />
      <span className="absolute bottom-1 left-2 font-mono text-xs text-ink-2">open</span>
      <span className="absolute bottom-1 right-2 font-mono text-xs text-ink-2">close</span>
    </div>
  );
}

/* ---------------------------------------------------------------- lane */

function FillProgress({ label, filled, target }: { label: string; filled: number; target: number }) {
  return (
    <span className="inline-flex items-center gap-1.5" title={`${label}: ${filled} of ${target} contracts filled`}>
      <span className="text-ink-3">{label}</span>
      <span className="h-1.5 w-12 overflow-hidden rounded-full bg-track" aria-hidden="true">
        <span className="block h-full rounded-full bg-blue" style={{ width: `${pctOf(filled, target)}%` }} />
      </span>
      <span className="tabular-nums">{Number.isInteger(filled) ? filled : filled.toFixed(2)} / {target}</span>
    </span>
  );
}

function Lane({ slot, index, now, selected }: { slot: Slot; index: number; now: number; selected: boolean }) {
  const phase = slotPhase(slot);
  const meta = STATE_TONE[phase] ?? { label: phase, tone: "inactive" as const };
  const edgeCents = Number(slot.edge) * 100;
  const live = Boolean(slot.order) || (phase === "momentum_burst" || phase === "mean_revert");
  const seriesData = slot.tape.length > 2
    ? slot.tape.map((p, i) => ({ x: i, p: p.p, mid: p.mid }))
    : [{ x: 0, p: 0.5, mid: 0.5 }, { x: 1, p: 0.5, mid: 0.5 }];

  return (
    <motion.div
      id={`lane-${slot.id}`}
      data-answers={index === 0 ? "2" : undefined}
      className={cn(
        "group relative grid w-full scroll-mt-24 grid-cols-1 items-center gap-3 rounded-xl px-2 py-3 text-left transition-colors duration-(--dur-swap) ease-(--ease-house) hover:bg-board md:grid-cols-[minmax(150px,1.1fr)_minmax(0,3fr)_auto] md:gap-5",
        selected && "bg-blue-soft ring-1 ring-blue/30"
      )}
    >
      {live && <BorderTrail size={70} className="bg-blue/70" />}
      <div className="flex min-w-0 items-center gap-3">
        <span className="font-mono text-xs text-ink-3">{String(index + 1).padStart(2, "0")}</span>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="truncate text-ui font-medium">{slot.display}</span>
            <Badge className={cn("h-[22px] rounded-full px-2.5 text-xs", TONE_CLASS[meta.tone])}>{meta.label}</Badge>
          </div>
          <div className="truncate font-mono text-xs text-ink-2">{slot.ticker ?? slot.series}</div>
        </div>
      </div>

      <div className="flex min-w-0 flex-col gap-2">
        <WindowRail slot={slot} now={now} />
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-xs text-ink-2">
          <span className="inline-flex items-center gap-1"><span className="text-ink-3">bid</span> {slot.yes_bid ? cents(slot.yes_bid) : "—"}</span>
          <span className="inline-flex items-center gap-1"><span className="text-ink-3">ask</span> {slot.yes_ask ? cents(slot.yes_ask) : "—"}</span>
          <span className="inline-flex items-center gap-1"><span className="text-ink-3">fair</span> {pct(slot.p_up)}</span>
          <span className="inline-flex items-center gap-1"><span className="text-ink-3">spot</span> {slot.spot ? slot.spot.toFixed(slot.spot > 100 ? 0 : 4) : "—"}</span>
          <span className="inline-flex items-center gap-1 text-ink-3">{slot.reason}</span>
          {slot.max_window_trades ? (
            <span
              className={cn("inline-flex items-center gap-1 rounded-chip px-1.5 py-0.5",
                (slot.window_entries ?? 0) >= slot.max_window_trades ? "bg-pill-warn-bg text-pill-warn-fg" : "bg-track text-ink-2")}
              title="Entries used this window against the per-window cap"
            >
              cap {slot.window_entries ?? 0} / {slot.max_window_trades}
            </span>
          ) : null}
          {slot.order ? (
            <FillProgress label="entry" filled={finite(slot.order.filled)} target={slot.order.contracts} />
          ) : null}
          {slot.exit ? (
            <FillProgress label="exit" filled={finite(slot.exit.filled)} target={slot.exit.contracts} />
          ) : null}
          {slot.signal_detail ? <span className="inline-flex items-center gap-1 tabular-nums">{slot.signal_detail}</span> : null}
          {slot.signal ? (
            <span
              className={cn("inline-flex items-center gap-1 rounded-chip px-1.5 py-0.5",
                slot.signal === "gbm" ? "bg-track text-ink-2" : "bg-pill-active-bg text-pill-active-fg")}
              title={slot.signal_detail}
            >
              {slot.signal}
            </span>
          ) : null}
          {slot.gate ? (
            <span
              className={cn("inline-flex items-center gap-1 rounded-chip px-1.5 py-0.5",
                slot.gate.available
                  ? (slot.gate.allow ? "bg-pill-active-bg text-pill-active-fg" : "bg-pill-warn-bg text-pill-warn-fg")
                  : "bg-track text-ink-2")}
              title={slot.gate.note}
            >
              laya {slot.gate.available ? (slot.gate.allow ? "enter" : "skip") : "offline"}
            </span>
          ) : null}
        </div>
      </div>

      <div className="flex items-center justify-between gap-4 md:justify-end">
        <div className="h-8 w-24 shrink-0">
          <SparkLineChart data={seriesData} categories={["p", "mid"]} index="x" colors={["blue", "gray"]} className="h-8 w-full" />
        </div>
        <div className="w-20 text-right">
          <div className={cn("font-mono text-ui font-medium tabular-nums", edgeCents > 0 ? "text-success-foreground" : edgeCents < 0 ? "text-destructive-foreground" : "text-ink-3")}>
            {edgeCents > 0 ? "+" : ""}{edgeCents.toFixed(1)}¢
          </div>
          <div className="font-mono text-xs text-ink-3">{mmss(slot.secs_left)}</div>
        </div>
      </div>
    </motion.div>
  );
}

/* ---------------------------------------------------------------- detail */

function DetailBody({ slot }: { slot: Slot }) {
  const p = slot.params;
  const tape = slot.tape.length > 2 ? slot.tape.map((t, i) => ({ x: i, fair: t.p, book: t.mid })) : [];
  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          ["fair", pct(slot.p_up)],
          ["book mid", slot.yes_bid && slot.yes_ask ? pct((Number(slot.yes_bid) + Number(slot.yes_ask)) / 2) : "—"],
          ["edge", `${(Number(slot.edge) * 100).toFixed(1)}¢`],
          ["confidence", pct(slot.confidence)],
        ].map(([label, value]) => (
          <div key={label} className="rounded-xl border border-hairline bg-board px-3 py-2">
            <div className="text-xs text-ink-3">{label}</div>
            <div className="font-mono text-ui font-medium">{value}</div>
          </div>
        ))}
      </div>

      <div>
        <div className="mb-2 flex items-center justify-between text-xs text-ink-2">
          <span>Fair value against the book, this window</span>
          <span className="font-mono">{slot.tape.length} samples</span>
        </div>
        {tape.length > 2 ? (
          <SparkAreaChart data={tape} categories={["fair", "book"]} index="x" colors={["blue", "gray"]} fill="none" className="h-28 w-full" />
        ) : (
          <p className="rounded-xl border border-dashed border-hairline p-4 text-sm text-ink-2">
            The tape fills once the spot model has a view; a quiet feed is not a signal.
          </p>
        )}
      </div>

      {slot.position && (
        <div className="rounded-xl border border-hairline p-3">
          <div className="mb-2 text-xs uppercase tracking-label text-ink-3">Open position</div>
          <div className="grid grid-cols-2 gap-2 font-mono text-sm sm:grid-cols-4">
            <div>{slot.position.side} · {slot.position.contracts} ct</div>
            <div>in {cents(slot.position.entry)}</div>
            <div>tp {cents(slot.position.tp)}</div>
            <div>sl {cents(slot.position.sl)}</div>
          </div>
        </div>
      )}
      {slot.order && !slot.position && (
        <div className="rounded-xl border border-hairline p-3">
          <div className="mb-2 text-xs uppercase tracking-label text-ink-3">Resting maker quote</div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-sm">
            <span>{slot.order.side} {slot.order.contracts} ct at {cents(slot.order.price)} · {slot.order.age_s.toFixed(0)}s</span>
            <FillProgress label="filled" filled={finite(slot.order.filled)} target={slot.order.contracts} />
          </div>
        </div>
      )}
      {slot.exit && (
        <div className="rounded-xl border border-hairline p-3">
          <div className="mb-2 text-xs uppercase tracking-label text-ink-3">Resting exit</div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-sm">
            <span>{slot.exit.contracts} ct at {cents(slot.exit.price)} · {slot.exit.reason}</span>
            <FillProgress label="filled" filled={finite(slot.exit.filled)} target={slot.exit.contracts} />
          </div>
        </div>
      )}

      <div>
        <div className="mb-2 text-xs uppercase tracking-label text-ink-3">Evolve block (live values)</div>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-xs text-ink-2 sm:grid-cols-3">
          {Object.entries(p).map(([k, v]) => (
            <div key={k} className="flex justify-between gap-2 border-b border-hairline py-1">
              <dt className="text-ink-3">{k}</dt>
              <dd className="tabular-nums">{String(v)}</dd>
            </div>
          ))}
        </dl>
      </div>
      <p className="text-sm text-ink-2">{slot.reason}</p>
    </div>
  );
}

function SlotDetail({ slot }: { slot: Slot }) {
  const reduce = useReducedMotion();
  const wide = useMediaQuery("(min-width: 768px)");
  const [open, setOpen] = useState(false);
  const label = `${slot.display} ${slot.ticker ?? slot.series}`;

  const body = (
    <div className="flex flex-col gap-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-title font-medium">{slot.display}</h2>
          <p className="font-mono text-xs text-ink-2">{slot.ticker ?? "no contract · " + slot.series}</p>
        </div>
        <Badge className={cn("h-[22px] rounded-full px-2.5 text-xs", TONE_CLASS[(STATE_TONE[slotPhase(slot)] ?? { tone: "inactive" as const }).tone])}>
          {(STATE_TONE[slotPhase(slot)] ?? { label: slot.state }).label}
        </Badge>
      </div>
      <DetailBody slot={slot} />
    </div>
  );

  if (!wide) {
    return (
      <>
        <Button variant="outline" size="sm" onClick={() => setOpen(true)} aria-label={`Open ${label}`}>Inspect</Button>
        <Drawer.Root open={open} onOpenChange={setOpen}>
          <Drawer.Portal>
            <Drawer.Overlay style={{ position: "fixed", inset: 0, zIndex: 40 }} className="bg-ink/25" />
            <Drawer.Content style={{ position: "fixed", left: 0, right: 0, bottom: 0, zIndex: 50 }} className="max-h-[86vh] overflow-y-auto rounded-t-dialog border-t border-hairline bg-surface p-5 pb-8">
              <Drawer.Handle className="mx-auto mb-4 h-1.5 w-10 rounded-full bg-track" />
              {body}
            </Drawer.Content>
          </Drawer.Portal>
        </Drawer.Root>
      </>
    );
  }

  return (
    <MorphingDialog transition={reduce ? { duration: 0 } : glide}>
      <MorphingDialogTrigger
        aria-label={`Open ${label}`}
        className="rounded-full text-xs font-medium text-blue outline-none transition-colors duration-(--dur-press) ease-(--ease-house) hover:text-blue-hover focus-visible:ring-2 focus-visible:ring-blue motion-safe:transition-[color,transform] motion-safe:active:scale-[0.96]"
      >
        <span className="px-2 py-1">Inspect</span>
      </MorphingDialogTrigger>
      <MorphingDialogContainer>
        <MorphingDialogContent disableLayoutAnimation={reduce === true} className="relative max-h-[86vh] w-[min(680px,92vw)] overflow-y-auto rounded-dialog border border-hairline bg-surface p-6 shadow-lift">
          <MorphingDialogTitle className="sr-only">{slot.display}</MorphingDialogTitle>
          <MorphingDialogDescription disableLayoutAnimation>{body}</MorphingDialogDescription>
          <MorphingDialogClose className="text-ink-2 hover:text-ink" />
        </MorphingDialogContent>
      </MorphingDialogContainer>
    </MorphingDialog>
  );
}

/* ---------------------------------------------------------------- panels */

function ModeBadge({ state }: { state: Snapshot }) {
  const armed = state.live_armed;
  return (
    <Badge className={cn("h-[22px] gap-1.5 rounded-full px-2.5 text-xs", armed ? "bg-pill-warn-bg text-pill-warn-fg" : "bg-tag-neutral-bg text-tag-neutral-fg")}>
      <span className={cn("inline-block size-1.5 rounded-full", armed ? "bg-dot-amber" : "bg-dot-green")} />
      {state.mode}
    </Badge>
  );
}


/* ---------------------------------------------------------------- readiness */

function CheckDot({ ok }: { ok: boolean }) {
  return (
    <span
      aria-hidden
      className={cn("mt-1.5 inline-block size-1.5 shrink-0 rounded-full", ok ? "bg-dot-green" : "bg-dot-amber")}
    />
  );
}

function LiveGate({ state }: { state: Snapshot }) {
  const r = state.readiness;
  const checks = r.checks ?? [];
  const blockers = r.blockers ?? [];
  const elapsed = r.elapsed_minutes ?? 0;
  const settled = r.settled_trades ?? 0;
  const passed = checks.filter((c) => c.ok).length;
  const checkPct = checks.length ? Math.min(100, (passed / checks.length) * 100) : 0;
  const gate = state.gate?.last;
  const live = state.live_armed;
  const verdict = (r.strategy ?? null) as
    | { promoted?: boolean; candidate?: { name?: string } | null; reason?: string }
    | null;
  const strategyName = verdict?.candidate?.name ?? "none";
  const strategyState = !verdict ? "unloaded" : verdict.promoted ? "promoted" : "rejected";
  const strategyDetail = verdict?.reason ?? "no verdict on disk";

  return (
    <section aria-label="Live arming" className="grid gap-4 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
      <div className="rounded-panel border border-hairline bg-surface p-4 sm:p-5">
        <header className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <ShieldCheck className="size-4 text-icon" strokeWidth={1.5} />
            <h2 className="text-ui font-medium">Arming to live</h2>
            <Badge className={cn("h-[22px] rounded-full px-2.5 text-xs", live ? "bg-pill-warn-bg text-pill-warn-fg" : "bg-tag-neutral-bg text-tag-neutral-fg")}>
              {live ? "armed" : "paper"}
            </Badge>
          </div>
          <span className="font-mono text-xs text-ink-3">
            {passed}/{checks.length} checks · venue {usd(state.venue_balance)}
          </span>
        </header>

        <div className="mt-3 flex flex-col gap-2">
          <div>
            <div className="flex items-baseline justify-between font-mono text-xs text-ink-2">
              <span>functional checks</span>
              <span className="tabular-nums">{passed} / {checks.length} passed</span>
            </div>
            <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-track">
              <motion.div className="h-full rounded-full bg-blue" animate={{ width: `${checkPct}%` }} transition={reveal} />
            </div>
          </div>
          <div className="flex items-baseline justify-between gap-3 font-mono text-xs text-ink-2">
            <span>this epoch</span>
            <span className="tabular-nums">{elapsed.toFixed(1)} min · {settled} settled · net {usd(r.net || "0")}</span>
          </div>
          <div className="flex items-baseline justify-between gap-3 font-mono text-xs text-ink-2">
            <span>strategy</span>
            <span className="truncate text-right tabular-nums" title={strategyDetail}>
              {strategyName} · {strategyState}
            </span>
          </div>
        </div>

        <ul className="mt-4 flex flex-col gap-2">
          {checks.map((c) => (
            <li key={c.key} className="flex items-start gap-2">
              <CheckDot ok={c.ok} />
              <div className="min-w-0">
                <div className={cn("text-sm", c.ok ? "text-ink" : "text-ink-2")}>{c.label}</div>
                <div className="font-mono text-xs text-ink-3">{c.detail}</div>
              </div>
            </li>
          ))}
        </ul>

        <p className="mt-3 rounded-xl bg-board px-3 py-2 text-xs text-ink-2">
          {r.ready
            ? "Every gate is green. The engine arms itself on the next pass and sends venue orders."
            : `Blocked by ${blockers.length}: ${blockers.slice(0, 3).join("; ") || "no reason reported"}.`}
          {r.kill_reason ? ` Kill switch: ${r.kill_reason}.` : ""}
        </p>
      </div>

      <div data-spacing="pinned" className="flex flex-col rounded-panel border border-hairline bg-surface p-4 sm:p-5">
        <header className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Brain className="size-4 text-icon" strokeWidth={1.5} />
            <h2 className="text-ui font-medium">Laya gate</h2>
          </div>
          <span className="inline-flex items-center gap-1.5 font-mono text-xs text-ink-2">
            <span className={cn("inline-block size-1.5 rounded-full", gate?.available ? "bg-dot-green" : "bg-dot-amber")} />
            {gate?.available ? gate.model : "offline"}
          </span>
        </header>
        <p className="mt-2 text-sm text-ink-2">
          {state.gate?.mode === "shadow"
            ? "Laya is in shadow: its probability and lane verdict are logged on every pass, but the deterministic kernel keeps the hot path until a fine-tuned model answers in under 35ms. A stale 1.7s answer is not an entry gate."
            : "Every entry is reviewed by Laya before a bid is sent. A veto is an abstention, and its reason shows on the lane."}
        </p>
        <div className="mt-3 flex flex-col divide-y divide-hairline">
          {[
            ["verdict", gate ? (gate.available ? (gate.allow ? "enter" : "skip") : "offline") : "—"],
            ["P(enter)", gate && gate.available ? gate.confidence.toFixed(3) : "—"],
            ["latency", gate ? `${gate.latency_ms.toFixed(0)} ms` : "—"],
            ["reviews", `${state.gate?.calls ?? 0} ok · ${state.gate?.failures ?? 0} failed`],
            ["mode", state.gate?.mode ?? "—"],
            ["settlement feed", state.ws ? `${state.ws.indices ?? 0} idx · ${(state.ws.messages ?? 0).toLocaleString()} msg` : "—"],
            ["ws lag", state.ws ? `${(state.ws.lag_ms ?? 0).toFixed(0)} ms · ${state.ws.sync_problems ?? 0} gaps` : "—"],
          ].map(([k, v]) => (
            <div key={k} className="flex items-center justify-between gap-3 py-2">
              <span className="text-sm text-ink-2">{k}</span>
              <span className="font-mono text-ui tabular-nums">{v}</span>
            </div>
          ))}
        </div>
        {gate?.error ? <p className="mt-2 font-mono text-xs text-pill-warn-fg">{gate.error}</p> : null}
        <p className="mt-auto pt-3 text-xs text-ink-3">
          The kernel still computes fair value, spread, fees and sizing; Laya only says whether this entry is worth taking.
        </p>
      </div>
    </section>
  );
}

function AccountPanel({ state }: { state: Snapshot }) {
  const day = Number(state.account.day_pnl);
  const curve = state.curve.length > 1 ? state.curve.map((p, i) => ({ x: i, v: p.equity })) : [];
  const since = state.epoch_start ?? 0;
  const epochTrades = state.trades.filter((t) => t.closed_ts >= since);
  const trades = epochTrades.length;
  const wins = epochTrades.filter((t) => Number(t.pnl) > 0).length;
  const start = Number(state.account.start_equity);
  const epochPnl = Number(state.account.equity) - start;
  const rec = state.reconcile ?? null;
  const delta = rec ? Number(rec.delta) : 0;
  const clean = rec !== null && Math.abs(delta) < 0.005;

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
      <Tile label="Equity, live" chip={
        <Badge
          className={cn("h-[22px] rounded-full px-2.5 font-mono text-xs", TONE_CLASS[rec === null ? "inactive" : clean ? "active" : "warn"])}
          title={rec ? `Venue ${usd(rec.venue_net)} against ledger ${usd(rec.ledger_net)}, checked ${rec.generated}` : "No venue reconciliation has run for this epoch yet"}
        >
          {rec === null ? "not reconciled yet" : `venue delta ${usd(delta)}`}
        </Badge>
      } className="rounded-panel">
        <div className="flex items-end justify-between gap-3">
          <Figure value={Number(state.account.equity)} prefix="$" size="md" group="," />
          <span className={cn("inline-flex items-center gap-1 font-mono text-sm", day >= 0 ? "text-success-foreground" : "text-destructive-foreground")}>
            {day >= 0 ? <ArrowUpRight className="size-4" strokeWidth={1.5} /> : <ArrowDownRight className="size-4" strokeWidth={1.5} />}
            {usd(day)}
            <Unit>today</Unit>
          </span>
        </div>
        {curve.length > 1 ? (
          <SparkAreaChart data={curve} categories={["v"]} index="x" colors={["blue"]} fill="none" className="h-60 w-full" />
        ) : (
          <div className="flex h-60 items-center justify-center rounded-xl border border-dashed border-hairline text-sm text-ink-2">
            The curve draws at the first marked position.
          </div>
        )}
        <p className="text-xs text-ink-3">
          Epoch start {usd(state.account.start_equity)} · slot sizing is 100% of a slot, never of the account.
        </p>
      </Tile>

      <section aria-label="Account detail" className="flex flex-col divide-y divide-hairline rounded-panel border border-hairline bg-surface px-4">
        {[
          ["Epoch P&L", usd(epochPnl)],
          ["Epoch ROI", start > 0 ? `${epochPnl >= 0 ? "+" : "−"}${Math.abs((epochPnl / start) * 100).toFixed(2)}%` : "—"],
          ["Realized", usd(state.account.realized)],
          ["Free cash", usd(state.account.cash)],
          ["Peak equity", usd(state.account.peak_equity)],
          ["Open slots", `${state.capacity?.open ?? state.account.open_positions} / ${state.capacity?.max ?? state.slots.length}`],
          ["Trades", `${trades}`],
          ["Win rate", trades ? pct(wins / trades) : "—"],
        ].map(([k, v]) => (
          <div key={k} className="flex items-center justify-between gap-3 py-3">
            <span className="text-sm text-ink-2">{k}</span>
            <span className="font-mono text-ui tabular-nums">{v}</span>
          </div>
        ))}
      </section>
    </div>
  );
}

function TapeLog({ state, query }: { state: Snapshot; query: string }) {
  const rows = state.fills.filter((f) =>
    !query || f.series.toLowerCase().includes(query) || f.reason.toLowerCase().includes(query)
  );
  return (
    <section aria-label="Tape log" className="rounded-panel border border-hairline bg-surface">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-hairline px-4 py-3">
        <div className="flex items-center gap-2">
          <Waves className="size-4 text-icon" strokeWidth={1.5} />
          <h2 className="text-ui font-medium">Tape log</h2>
          <span className="inline-flex h-5 items-center rounded-chip bg-track px-1.5 font-mono text-xs text-ink-2">{rows.length}</span>
        </div>
        <span className="text-xs text-ink-3">maker quotes are free; every exit shows its fee</span>
      </header>
      {rows.length === 0 ? (
        <p className="px-4 py-8 text-sm text-ink-2">
          No fills yet. The engine only quotes when the model clears the spread, the band and the fee by 1.5¢.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-ink-3">
                <th className="hidden px-4 py-2 font-medium sm:table-cell">Time</th>
                <th className="px-4 py-2 font-medium">Slot</th>
                <th className="hidden px-4 py-2 font-medium sm:table-cell">Side</th>
                <th className="px-4 py-2 font-medium">Kind</th>
                <th className="px-4 py-2 font-medium">Price</th>
                <th className="hidden px-4 py-2 font-medium sm:table-cell">Size</th>
                <th className="hidden px-4 py-2 font-medium sm:table-cell">Fee</th>
                <th className="px-4 py-2 font-medium">PnL</th>
                <th className="hidden px-4 py-2 font-medium md:table-cell">Why</th>
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, 12).map((f) => (
                <tr key={`${f.ts}-${f.kind}-${f.side}`} className="border-t border-hairline">
                  <td className="hidden px-4 py-2 font-mono text-xs text-ink-2 sm:table-cell">{clock(f.ts)}</td>
                  <td className="px-4 py-2 font-mono text-xs">{f.series.replace("KX", "").replace("15M", "")}</td>
                  <td className="hidden px-4 py-2 font-mono text-xs sm:table-cell">{f.side}</td>
                  <td className="px-4 py-2">
                    <Badge variant="outline" className="rounded-full px-2 text-xs">{f.kind}</Badge>
                  </td>
                  <td className="px-4 py-2 font-mono text-xs">{cents(f.price)}</td>
                  <td className="hidden px-4 py-2 font-mono text-xs sm:table-cell">{f.contracts}</td>
                  <td className="hidden px-4 py-2 font-mono text-xs sm:table-cell">
                    {(f.kind === "entry" || f.kind === "exit") && feeSide(f.fee) ? (
                      <span className={feeSide(f.fee)!.fee > 0 ? "text-pill-warn-fg" : "text-ink-2"}>
                        {feeSide(f.fee)!.label} {usd(feeSide(f.fee)!.fee)}
                      </span>
                    ) : <span className="text-ink-3">—</span>}
                  </td>
                  <td className={cn("px-4 py-2 font-mono text-xs", Number(f.pnl) > 0 ? "text-success-foreground" : Number(f.pnl) < 0 ? "text-destructive-foreground" : "text-ink-3")}>
                    {Number(f.pnl) ? usd(f.pnl) : "—"}
                  </td>
                  <td className="hidden px-4 py-2 text-xs text-ink-2 md:table-cell">{f.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

const LOOP_STATES = ["RUNNING", "HALT_FLATTEN", "CAPTURE", "PROPOSE", "VERIFY", "SWAP", "ABSTAIN"];

function LoopPanel({ state }: { state: Snapshot }) {
  const shr = state.shr;
  const current = shr.state;
  const ece = state.slots.length ? Math.max(...state.slots.map((s) => s.ece)) : 0;
  const outcomes = state.trades.slice(0, 24).reverse().map((t, i) => ({
    key: i,
    color: Number(t.pnl) > 0 ? "bg-dot-green" : Number(t.pnl) < 0 ? "bg-unread" : "bg-dot-amber",
    tooltip: `${clock(t.closed_ts)} · ${t.series} · ${usd(t.pnl)}`,
  }));
  const [openFault, setOpenFault] = useState(false);

  return (
    <section aria-label="Recursive loop" className="min-w-0 overflow-hidden rounded-panel border border-hairline bg-surface">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-hairline px-4 py-3">
        <div className="flex items-center gap-2">
          <Activity className="size-4 text-icon" strokeWidth={1.5} />
          <h2 className="text-ui font-medium">Recursive loop</h2>
          <Pill className="h-[22px] gap-1.5 px-2 text-xs">
            <PillIndicator variant={current === "RUNNING" ? "success" : current === "ABSTAIN" ? "warning" : "info"} pulse={current !== "RUNNING"} />
            <span className="font-mono text-xs">{current}</span>
          </Pill>
        </div>
        <div className="flex items-center gap-4 font-mono text-xs text-ink-2">
          <span>gen {shr.generation}</span>
          <span className="text-success-foreground">{shr.promoted} promoted</span>
          <span className="text-ink-3">{shr.rejected} rejected</span>
        </div>
      </header>

      <div className="grid min-w-0 gap-5 px-4 py-4 lg:grid-cols-[minmax(0,1fr)_auto]">
        <div className="flex min-w-0 flex-col gap-4">
          <ol className="flex flex-wrap items-center gap-1.5">
            {LOOP_STATES.map((s) => (
              <li key={s}>
                <Badge
                  className={cn(
                    "h-[22px] rounded-full px-2.5 font-mono text-xs transition-colors duration-(--dur-swap)",
                    s === current ? "bg-blue text-blue-fg" : "bg-track text-ink-2"
                  )}
                >
                  {s}
                </Badge>
              </li>
            ))}
          </ol>

          <div>
            <div className="mb-2 flex items-center justify-between text-xs text-ink-2">
              <span>Last {outcomes.length || 0} closed trades</span>
              <span className="font-mono">{outcomes.filter((o) => o.color === "bg-dot-green").length} green</span>
            </div>
            {outcomes.length ? (
              <Tracker data={outcomes} className="h-9" />
            ) : (
              <p className="rounded-xl border border-dashed border-hairline px-3 py-4 text-sm text-ink-2">
                No settled trades to grade yet.
              </p>
            )}
          </div>

          <Expandable expanded={openFault} onToggle={() => setOpenFault((v) => !v)} transitionDuration={0.2}>
            <ExpandableTrigger className="flex w-full items-center justify-between gap-2 rounded-xl border border-hairline bg-board px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-blue">
              <span className="inline-flex items-center gap-2">
                <ShieldAlert className="size-4 text-icon" strokeWidth={1.5} />
                Forensic slice {shr.fault ? "captured" : "idle"}
              </span>
              <span className="font-mono text-xs text-ink-3">{shr.fault ? `${shr.fault.ticks} ticks` : "nothing to audit"}</span>
            </ExpandableTrigger>
            <ExpandableContent preset="slide-down">
              <div className="px-3 pt-2 text-sm text-ink-2">
                {shr.fault ? (
                  <p>
                    <span className="font-medium text-ink">{shr.fault.reason}</span> at {clock(shr.fault.ts)}. The slot flattened,
                    kept its capital and handed the exact ticks to the verifier. The program database below holds every candidate
                    it scored.
                  </p>
                ) : (
                  <p>No halt has fired. A slot stops on three straight losses, a 12% drawdown from its own high-water mark, or ECE above 0.08.</p>
                )}
              </div>
            </ExpandableContent>
          </Expandable>

          {shr.timeline.length > 0 && (
            <ul className="flex flex-col gap-1 font-mono text-xs text-ink-2">
              {shr.timeline.slice(-4).reverse().map((t) => (
                <li key={`${t.ts}-${t.state}`} className="flex min-w-0 gap-2">
                  <span className="text-ink-3">{clock(t.ts)}</span>
                  <span className="text-blue">{t.state}</span>
                  <span className="min-w-0 truncate">{t.detail}</span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="flex min-w-0 items-center justify-center gap-4 rounded-xl border border-hairline bg-board px-4 py-3">
          <ProgressCircle value={Math.round((1 - Math.min(1, ece)) * 100)} radius={40} strokeWidth={8}>
            <span className="font-mono text-sm font-medium">{Math.round((1 - Math.min(1, ece)) * 100)}%</span>
          </ProgressCircle>
          <div className="min-w-0 text-sm">
            <div className="font-medium">Calibration</div>
            <div className="font-mono text-xs text-ink-2">ECE {ece.toFixed(3)}</div>
            <div className="mt-1 text-xs text-ink-3">Above 0.08 the slot halts itself.</div>
          </div>
        </div>
      </div>

      <div className="grid gap-4 border-t border-hairline px-4 py-4 lg:grid-cols-2">
        <div>
          <div className="mb-2 text-xs uppercase tracking-label text-ink-3">Candidates this halt</div>
          {shr.candidates.length === 0 ? (
            <p className="text-sm text-ink-2">The proposer only runs after a stop. Nothing is being mutated now.</p>
          ) : (
            <ul className="flex flex-col gap-2">
              {shr.candidates.map((c) => (
                <motion.li layout key={c.name} transition={layout} className="rounded-xl border border-hairline px-3 py-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-medium">{c.name}</span>
                    <Badge
                      className={cn(
                        "h-[22px] rounded-full px-2 text-xs",
                        c.status === "promoted" ? "bg-pill-active-bg text-pill-active-fg" : c.status === "verified" ? "bg-tag-ops-bg text-tag-ops-fg" : "bg-pill-inactive-bg text-pill-inactive-fg"
                      )}
                    >
                      {c.status}
                    </Badge>
                  </div>
                  <div className="mt-1 flex gap-4 font-mono text-xs text-ink-2">
                    <span>{usd(c.score)}</span>
                    <span>PF {c.profit_factor.toFixed(2)}</span>
                    <span>ECE {c.ece.toFixed(3)}</span>
                  </div>
                  <p className="mt-1 text-xs text-ink-3">{c.note}</p>
                </motion.li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <div className="mb-2 text-xs uppercase tracking-label text-ink-3">Program database</div>
          {shr.programs.length === 0 ? (
            <p className="text-sm text-ink-2">Empty until the first verifier run. Every candidate is written here with its score.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-ink-3">
                    {["Gen", "Slot", "Name", "Status", "Score"].map((h) => <th key={h} className="py-1 pr-3 font-medium">{h}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {shr.programs.slice(0, 8).map((p) => (
                    <tr key={p.id} className="border-t border-hairline">
                      <td className="py-1.5 pr-3 font-mono text-xs">{p.generation}</td>
                      <td className="py-1.5 pr-3 font-mono text-xs">{p.slot.replace("KX", "").replace("15M", "")}</td>
                      <td className="py-1.5 pr-3 text-xs">{p.name}</td>
                      <td className="py-1.5 pr-3 text-xs text-ink-2">{p.status}</td>
                      <td className="py-1.5 pr-3 font-mono text-xs">{usd(p.score)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- app */

function Console({ state }: { state: Snapshot }) {
  const { theme, toggle } = useTheme();
  const [query, setQuery] = useState("");
  const [focus, setFocus] = useState<string | null>(null);
  const now = state.generated_at;
  const q = query.trim().toLowerCase();

  const slots = useMemo(
    () => state.slots.filter((s) => !q || s.asset.toLowerCase().includes(q) || s.display.toLowerCase().includes(q) || s.state.toLowerCase().includes(q)),
    [state.slots, q]
  );
  const openSecs = Math.min(...state.slots.map((s) => s.secs_left).filter((v) => v > 0), 900);
  const openTs = state.window.open_ts || state.slots[0]?.window_open_ts || now - 900;
  const closeTs = state.window.close_ts || openTs + 900;
  const elapsedPct = pctOf(now - openTs, closeTs - openTs);
  const healthy = state.health.coinbase && state.health.market_errors === 0;
  const working = state.slots.filter((s) => s.position || s.order).length;

  const lastFill = state.fills[0];
  const seenFill = useRef<number | null>(null);
  const seenState = useRef<string | null>(null);
  useEffect(() => {
    if (lastFill && seenFill.current !== null && lastFill.ts !== seenFill.current) {
      toast(`${lastFill.series.replace("KX", "").replace("15M", "")} ${lastFill.side} · ${lastFill.kind}`, {
        description: `${cents(lastFill.price)} · ${lastFill.contracts} contracts · ${lastFill.reason}`,
      });
    }
    if (lastFill) seenFill.current = lastFill.ts;
  }, [lastFill]);
  useEffect(() => {
    if (seenState.current !== null && seenState.current !== state.shr.state && state.shr.state !== "RUNNING") {
      toast(`Loop ${state.shr.state}`, { description: state.shr.timeline.at(-1)?.detail ?? "state changed" });
    }
    seenState.current = state.shr.state;
  }, [state.shr.state, state.shr.timeline]);

  const selectSlot = useCallback((id: string) => {
    setFocus(id);
    window.setTimeout(() => document.getElementById(`lane-${id}`)?.scrollIntoView({ behavior: "smooth", block: "center" }), 40);
  }, []);

  return (
    <div className="min-h-dvh bg-canvas px-3 py-4 sm:px-5 sm:py-6">
      <div className="mx-auto flex w-full max-w-[1400px] flex-col gap-4 rounded-shell border border-hairline bg-surface p-3 shadow-shell sm:p-5">
        <header className="flex flex-col gap-3 border-b border-hairline pb-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-xl bg-ink-button font-mono text-xs font-medium text-ink-button-fg">OK</div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-ui font-medium tracking-tight">Open-Kal</h1>
                <span data-answers="1" className="inline-flex"><ModeBadge state={state} /></span>
              </div>
              <p className="font-mono text-xs text-ink-2">15-minute Kalshi engine · {state.env}</p>
            </div>
          </div>

          <div className="flex min-w-0 flex-1 items-center gap-3 lg:max-w-xl">
            <Island
              inline
              sentence={
                healthy ? (
                  <>
                    <span className="font-mono">{state.health.venue_calls}</span> venue calls, <span className="font-mono">{working}</span> {working === 1 ? "slot" : "slots"} working this window
                  </>
                ) : (
                  <>Feed degraded. The engine abstains rather than guesses.</>
                )
              }
              attention={!healthy}
              event={null}
              noun="slots"
              query={query}
              onQuery={setQuery}
              items={slots.map((s) => ({ id: s.id, label: s.display, meta: s.ticker ?? s.state }))}
              commands={[
                { id: "theme", label: `Switch to ${theme === "dark" ? "light" : "dark"}`, Icon: theme === "dark" ? Sun : Moon, run: toggle },
                { id: "fills", label: "Go to the tape log", Icon: Terminal, run: () => document.getElementById("tape-log")?.scrollIntoView({ behavior: "smooth" }) },
                { id: "loop", label: "Go to the recursive loop", Icon: Gauge, run: () => document.getElementById("loop")?.scrollIntoView({ behavior: "smooth" }) },
                { id: "arming", label: "Go to live arming", Icon: ShieldCheck, run: () => document.getElementById("arming")?.scrollIntoView({ behavior: "smooth" }) },
              ]}
              onOpenItem={selectSlot}
            />
          </div>

          <div className="flex items-center gap-2">
            <span data-answers="5" className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-mono text-xs", state.shr.state === "RUNNING" ? "bg-pill-active-bg text-pill-active-fg" : "bg-pill-warn-bg text-pill-warn-fg")}>
              <span className={cn("inline-block size-1.5 rounded-full", state.shr.state === "RUNNING" ? "bg-dot-green" : "bg-dot-amber")} />
              loop {state.shr.state.toLowerCase()}
            </span>
            <span className="hidden items-center gap-2 text-xs text-ink-2 sm:inline-flex">
              <CircleDot className={cn("size-3.5", healthy ? "text-dot-green" : "text-dot-amber")} strokeWidth={2} />
              <span className="font-mono">{(state.health?.latency_ms ?? 0).toFixed(0)}ms</span>
            </span>
            {state.health.coinbase && state.health.kraken ? (
              <span className="hidden font-mono text-xs text-ink-2 md:inline">coinbase + kraken</span>
            ) : (
              <span className="hidden font-mono text-xs text-dot-amber md:inline">{state.health.coinbase ? "coinbase" : "no spot"}</span>
            )}
            <Button variant="outline" size="icon" onClick={toggle} aria-label="Switch theme">
              {theme === "dark" ? <Sun className="size-4" strokeWidth={1.5} /> : <Moon className="size-4" strokeWidth={1.5} />}
            </Button>
          </div>
        </header>

        <section aria-label="Current window" className="rounded-panel border border-hairline bg-surface p-3 sm:p-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 text-xs uppercase tracking-label text-ink-3">
                <Timer className="size-3.5" strokeWidth={1.5} />
                Window {clock(openTs)} → {clock(closeTs)} UTC
              </div>
              <h2 className="mt-1 text-title font-medium">Where the edge is, right now</h2>
              <div data-answers="3" className="mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <span className="font-mono text-ui font-medium tabular-nums">{usd(state.account.equity)}</span>
                <span className={cn("font-mono text-xs", Number(state.account.day_pnl) >= 0 ? "text-success-foreground" : "text-destructive-foreground")}>
                  {Number(state.account.day_pnl) >= 0 ? "+" : ""}{usd(state.account.day_pnl)} today
                </span>
                <span className="text-xs text-ink-3">across {state.slots.length} slots</span>
              </div>
              <p className="mt-1 max-w-xl text-sm text-ink-2">
                Five orthogonal slots share one fifteen-minute clock. Each lane is the contract's own window, the bar is what the
                engine holds, and the marker is now.
              </p>
            </div>
            <div className="flex items-center gap-5">
              <div>
                <div className="text-xs text-ink-3">to close</div>
                <div className="font-mono text-stat font-medium tabular-nums">{mmss(openSecs)}</div>
              </div>
              <div className="w-40">
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-track">
                  <motion.div className="h-full rounded-full bg-blue" animate={{ width: `${elapsedPct}%` }} transition={reveal} />
                </div>
                <div className="mt-1 flex justify-between font-mono text-xs text-ink-3">
                  <span>{elapsedPct.toFixed(0)}%</span>
                  <span>{state.slots.length} slots</span>
                </div>
              </div>
            </div>
          </div>

          <div data-answers="4" className="mt-4 flex flex-wrap items-center gap-2 rounded-xl bg-board px-3 py-2 font-mono text-xs text-ink-2">
            <span className="text-ink-3">last action</span>
            {lastFill ? (
              <>
                <span className="text-ink">{lastFill.series.replace("KX", "").replace("15M", "")}</span>
                <span>{lastFill.side} {lastFill.kind}</span>
                <span>{cents(lastFill.price)}</span>
                <span className="text-ink-3">{lastFill.reason}</span>
                <span className="ml-auto text-ink-3">{clock(lastFill.ts)}</span>
              </>
            ) : (
              <span>no fills yet · the engine is quoting only inside its band</span>
            )}
          </div>

          <div className="mt-2 flex flex-col divide-y divide-hairline">
            {slots.map((s, i) => (
              <div key={s.id} className="flex items-center gap-2">
                <Lane slot={s} index={i} now={now} selected={focus === s.id} />
                <SlotDetail slot={s} />
              </div>
            ))}
            {slots.length === 0 && (
              <p className="py-6 text-sm text-ink-2">No slot matches “{query}”.</p>
            )}
          </div>
        </section>

        <AccountPanel state={state} />

        <div id="arming" className="scroll-mt-6">
          <LiveGate state={state} />
        </div>

        <div id="tape-log" className="scroll-mt-6">
          <TapeLog state={state} query={q} />
        </div>

        <div id="loop" className="scroll-mt-6">
          <LoopPanel state={state} />
        </div>

        <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-hairline pt-3 text-xs text-ink-3">
          <span className="inline-flex items-center gap-2">
            <TextShimmer className="inline-flex items-center rounded-chip bg-track px-1.5 py-1 font-mono" duration={2.4}>streaming</TextShimmer>
            <span>venue snapshots every 3s · spot websockets from Coinbase and Kraken</span>
          </span>
          <span className="font-mono">
            {state.live_armed
              ? "LIVE ORDERS · post-only on Kalshi production"
              : "paper fills · live orders stay off until every readiness check passes"}
          </span>
        </footer>
      </div>

      <Toaster position="bottom-right" toastOptions={{ className: "font-sans text-sm" }} />
    </div>
  );
}

function Shell() {
  const [state, setState] = useState<Snapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const res = await fetch("/api/state", { cache: "no-store" });
        const raw = (await res.json()) as Record<string, unknown> | null;
        if (!alive) return;
        if (!raw || typeof raw !== "object" || !("account" in raw)) { setError("engine is booting"); return; }
        setState(normalizeSnapshot(raw));
        setError(null);
      } catch {
        if (alive) setError("engine unreachable");
      }
    };
    load();
    const id = window.setInterval(load, 1000);
    return () => { alive = false; window.clearInterval(id); };
  }, []);

  if (!state) {
    return (
      <div className="flex min-h-dvh items-center justify-center bg-canvas p-6">
        <div className="w-full max-w-md rounded-panel border border-hairline bg-surface p-6 shadow-card">
          <div className="mb-2 flex items-center gap-2">
            <span className="flex size-8 items-center justify-center rounded-lg bg-ink-button font-mono text-xs text-ink-button-fg">OK</span>
            <h1 className="text-ui font-medium">Open-Kal</h1>
          </div>
          <p className="text-sm text-ink-2">
            {error ?? "Connecting to the engine on this machine."} Start it with{" "}
            <code className="rounded-chip bg-track px-1.5 py-0.5 font-mono text-xs">.venv/bin/python run.py</code>.
          </p>
        </div>
      </div>
    );
  }
  return <Console state={state} />;
}

export default function App() {
  return (
    <ThemeProvider>
      <Shell />
    </ThemeProvider>
  );
}
