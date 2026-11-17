// Move 3 and move 6: the instrument tile and the figure. A tile is one Tremor part read at a glance; a figure is
// Geist Mono and rolls when it changes (Motion Primitives sliding number, which exposes the value once to a screen reader).
import type { ReactNode } from "react";
import { SlidingNumber } from "@/components/ui/sliding-number";
import { cn } from "@/lib/utils";

const SIZE = { inline: "text-sm leading-5", sm: "text-stat", md: "text-[40px] leading-[44px]", hero: "text-[72px] leading-[72px]" } as const;

export function Figure({ value, prefix, suffix, size = "sm", group, className }: {
  value: number; prefix?: ReactNode; suffix?: ReactNode; size?: keyof typeof SIZE; group?: string; className?: string;
}) {
  return (
    <div data-figure className={cn("inline-flex items-baseline font-mono font-medium tracking-tight", SIZE[size], className)}>
      {prefix}
      <SlidingNumber value={value} group={group} />
      {suffix}
    </div>
  );
}

/** The small unit label that sits after a figure ("of 6", "run"). */
export function Unit({ children }: { children: ReactNode }) {
  return <span className="ml-1.5 font-sans text-sm font-normal text-ink-2">{children}</span>;
}

export function Chip({ children, tone = "active" }: { children: ReactNode; tone?: "active" | "warn" | "flat" }) {
  const t = { active: "bg-pill-active-bg text-pill-active-fg", warn: "bg-pill-warn-bg text-pill-warn-fg", flat: "bg-track text-ink-2" }[tone];
  return <span className={cn("inline-flex h-5 items-center rounded-full px-1.5 font-mono text-xs font-medium", t)}>{children}</span>;
}

export function Tile({ label, chip, id, className, children, ...rest }: {
  label: string; chip?: ReactNode; id?: string; className?: string; children: ReactNode; "data-tile"?: string;
}) {
  return (
    <section id={id} aria-label={label} {...rest} className={cn("flex min-w-0 scroll-mt-24 flex-col gap-3 rounded-2xl border border-hairline bg-surface p-4 sm:p-5", className)}>
      <header className="flex items-center justify-between gap-2 text-sm text-ink-2">
        <h2 className="truncate font-normal">{label}</h2>
        {chip}
      </header>
      {children}
    </section>
  );
}
