// Move 2: the island. One live sentence about the product, top centre, on Cult UI's dynamic island. It morphs to
// announce a system event and settles back, and it opens into search and commands (Ctrl+K or Cmd+K) with a COSS
// input and keyboard hints. The sentence belongs to the product; the island never carries navigation (the dock does).
import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { motion } from "motion/react";
import { CornerDownLeft, Search, type LucideIcon } from "lucide-react";
import DynamicIsland, { DynamicContainer, DynamicIslandProvider, SIZE_PRESETS, useDynamicIslandSize } from "@/components/ui/dynamic-island";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { Kbd } from "@/components/ui/kbd";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { reveal } from "@/motion";

export type IslandItem = { id: string; label: string; meta?: string };
export type IslandCommand = { id: string; label: string; Icon: LucideIcon; run: () => void };
export type IslandEvent = { id: number; text: string } | null;

export type IslandProps = {
  sentence: ReactNode;
  attention?: boolean;
  event: IslandEvent;
  /** What is searched, plural: "jobs", "entries", "posts". */
  noun: string;
  query: string;
  onQuery: (q: string) => void;
  /** The items that match the query (the study filters its own list with the same query). */
  items: IslandItem[];
  commands: IslandCommand[];
  onOpenItem: (id: string) => void;
  /** Sits in the flow of its parent instead of fixed to the top of the window (the bench uses this). */
  inline?: boolean;
};

function IslandBody({ sentence, attention, event, noun, query, onQuery, items, commands, onOpenItem, inline }: IslandProps) {
  const { dispatch } = useDynamicIslandSize();
  const [open, setOpen] = useState(false);
  const [flash, setFlash] = useState<IslandEvent>(null);
  const root = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const wasOpen = useRef(false);
  const q = query.trim().toLowerCase();
  const cmds = useMemo(() => commands.filter((c) => !q || c.label.toLowerCase().includes(q)), [commands, q]);

  // An event spoken by the system: the island grows, says it, and settles back.
  useEffect(() => {
    if (!event) return;
    setFlash(event);
    const t = setTimeout(() => setFlash(null), 3800);
    return () => clearTimeout(t);
  }, [event]);

  useEffect(() => {
    dispatch({ type: "SET_SIZE", newSize: open ? SIZE_PRESETS.PALETTE : flash ? SIZE_PRESETS.EVENT : SIZE_PRESETS.SENTENCE });
  }, [open, flash, dispatch]);

  // Ctrl+K or Cmd+K opens it from anywhere; the input takes focus. Closing gives focus back to the sentence.
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen(true);
        requestAnimationFrame(() => { input.current?.focus(); input.current?.select(); });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => {
    if (open) input.current?.focus();
    else if (wasOpen.current) trigger.current?.focus();
    wasOpen.current = open;
  }, [open]);
  useEffect(() => {
    if (!open) return;
    const away = (e: PointerEvent) => { if (!root.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("pointerdown", away);
    return () => document.removeEventListener("pointerdown", away);
  }, [open]);

  const rows = () => [...(root.current?.querySelectorAll<HTMLElement>("[data-island-row]") ?? [])];
  const pick = (id: string) => { setOpen(false); setTimeout(() => onOpenItem(id), 60); };
  const runCommand = (c: IslandCommand) => { setOpen(false); c.run(); };

  const onInputKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") { e.preventDefault(); rows()[0]?.focus(); }
    if (e.key === "Enter") {
      e.preventDefault(); // focus moves to the sentence key as the palette closes; without this the same Enter would press it and reopen the palette
      if (q && items[0]) pick(items[0].id);
      else if (q && cmds[0]) runCommand(cmds[0]);
    }
  };
  const onRowKey = (e: KeyboardEvent<HTMLElement>) => {
    const list = rows(); const i = list.indexOf(e.currentTarget);
    if (e.key === "ArrowDown") { e.preventDefault(); list[Math.min(i + 1, list.length - 1)]?.focus(); }
    if (e.key === "ArrowUp") { e.preventDefault(); (i === 0 ? input.current : list[i - 1])?.focus(); }
  };
  const onPaletteKey = (e: KeyboardEvent<HTMLElement>) => {
    if (e.key !== "Escape") return;
    e.stopPropagation();
    if (query) onQuery(""); else setOpen(false);
  };

  const row = "flex h-10 w-full cursor-pointer items-center gap-3 rounded-xl px-3 text-left text-sm outline-none hover:bg-ink-button-fg/10 focus-visible:bg-ink-button-fg/15 focus-visible:ring-2 focus-visible:ring-ink-button-fg/70";
  const label = "px-3 pb-1 pt-2 text-xs font-medium uppercase tracking-label text-ink-button-fg/70";

  return (
    <div className={inline ? "relative z-30 flex h-12 w-full items-start justify-center [&_#island]:max-w-full [&_div.z-10.items-end]:items-start" : "pointer-events-none fixed inset-x-0 top-3 z-50"}>
      <div ref={root} onKeyDown={open ? onPaletteKey : undefined} className="contents">
        <DynamicIsland id="island" {...(open ? { role: "dialog", "aria-label": "Search and commands" } : {})}>
          <DynamicContainer className="h-full w-full">
            {open ? (
              <div className="flex h-full flex-col text-left">
                <div className="p-2">
                  <InputGroup className="h-11 rounded-xl border-ink-button-fg/25 !bg-transparent">
                    <InputGroupAddon><Search className="size-4 text-icon" strokeWidth={1.5} aria-hidden /></InputGroupAddon>
                    <InputGroupInput
                      ref={input}
                      type="search"
                      placeholder={`Search ${noun}`}
                      aria-label={`Search ${noun}`}
                      className="[&_input]:text-ink-button-fg [&_input::placeholder]:text-ink-button-fg/70"
                      value={query}
                      onChange={(e) => onQuery(e.target.value)}
                      onKeyDown={onInputKey}
                    />
                    <InputGroupAddon align="inline-end"><Kbd>Esc</Kbd></InputGroupAddon>
                  </InputGroup>
                </div>
                <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
                  {q && !items.length && !cmds.length ? (
                    <div data-search-empty role="status" className="flex flex-col items-center gap-2 px-4 py-8 text-center">
                      <div className="max-w-full break-words text-ui font-medium">No {noun} match “{query.trim()}”</div>
                      <p className="text-sm text-ink-button-fg/70">The search looks at the name, the id and the category.</p>
                      <Button variant="outline" size="sm" onClick={() => onQuery("")}>Clear search</Button>
                    </div>
                  ) : (
                    <>
                      {items.length > 0 && (
                        <>
                          <div className={label}>{noun[0].toUpperCase() + noun.slice(1)}{q ? ` · ${items.length}` : ""}</div>
                          {(q ? items : items.slice(0, 4)).slice(0, 6).map((it) => (
                            <button key={it.id} type="button" data-island-row onKeyDown={onRowKey} onClick={() => pick(it.id)} className={row}>
                              <span className="font-mono text-xs text-ink-button-fg/70">{it.id}</span>
                              <span className="min-w-0 flex-1 truncate">{it.label}</span>
                              {it.meta && <span className="font-mono text-xs text-ink-button-fg/70">{it.meta}</span>}
                            </button>
                          ))}
                        </>
                      )}
                      {cmds.length > 0 && (
                        <>
                          <div className={label}>Commands</div>
                          {cmds.map((c) => (
                            <button key={c.id} type="button" data-island-row onKeyDown={onRowKey} onClick={() => runCommand(c)} className={row}>
                              <c.Icon className="size-4 shrink-0" strokeWidth={1.5} aria-hidden />
                              <span className="min-w-0 flex-1 truncate">{c.label}</span>
                            </button>
                          ))}
                        </>
                      )}
                    </>
                  )}
                </div>
                <div className="hidden items-center gap-4 border-t border-ink-button-fg/15 px-4 py-2 text-xs text-ink-button-fg/70 sm:flex">
                  <span className="inline-flex items-center gap-1.5"><Kbd>↑</Kbd><Kbd>↓</Kbd> move</span>
                  <span className="inline-flex items-center gap-1.5"><Kbd><CornerDownLeft aria-hidden /></Kbd> open</span>
                  <span className="inline-flex items-center gap-1.5"><Kbd>Esc</Kbd> clear, then close</span>
                </div>
              </div>
            ) : (
              <button
                ref={trigger}
                type="button"
                aria-haspopup="dialog"
                aria-expanded={false}
                aria-keyshortcuts="Control+K Meta+K"
                onClick={() => setOpen(true)}
                className="flex h-full w-full cursor-pointer items-center gap-2.5 px-4 text-left text-sm outline-none transition-colors duration-(--dur-press) ease-(--ease-house) hover:bg-ink-button-fg/10 active:bg-ink-button-fg/15 focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ink-button-fg/70"
              >
                <span aria-hidden className={cn("size-2 shrink-0 rounded-full", attention ? "bg-dot-clay" : "bg-dot-green")} />
                <motion.span key={flash ? `e${flash.id}` : query ? "q" : "s"} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={reveal} className="min-w-0 flex-1 truncate" data-island-text>
                  {flash ? flash.text : query ? <>Filtering for “{query}”</> : sentence}
                </motion.span>
                <Kbd className="hidden sm:inline-flex">⌘K</Kbd>
              </button>
            )}
          </DynamicContainer>
        </DynamicIsland>
      </div>
    </div>
  );
}

export function Island(props: IslandProps) {
  return (
    <DynamicIslandProvider initialSize={SIZE_PRESETS.SENTENCE}>
      <IslandBody {...props} />
    </DynamicIslandProvider>
  );
}
