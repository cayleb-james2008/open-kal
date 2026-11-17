# Role coverage — every library, and the need of this product it serves

| Library | Need of this product | Part used and its file | If "no need here": why |
|---|---|---|---|
| Motion Primitives | A slot opens into its own tape and parameters without leaving the window; figures must roll when money changes | `MorphingDialog*` and `SlidingNumber` in `src/App.tsx` (via `components/ui/morphing-dialog.tsx`, `components/ui/sliding-number.tsx`) | — |
| COSS UI | Dense controls and keys: theme switch, the inspector trigger, keyboard hints, hairline table | `Button`, `Kbd`, `InputGroup` wrappers inside `chrome/Island.tsx` | — |
| Cult UI | A halt's forensic detail opens in place, under the line that says it was captured | `Expandable`, `ExpandableContent`, `ExpandableTrigger` from `components/ui/expandable.tsx`, used in `LoopPanel` | — |
| Kibo UI | Slot and loop state need one chip vocabulary, with a live indicator while work runs | `Pill`, `PillIndicator` from `components/kibo-ui/pill/index.tsx` | — |
| Tremor | The equity curve, the per-lane fair-vs-book spark, the last-24-outcomes strip and the calibration ring | `SparkAreaChart`, `SparkLineChart`, `Tracker`, `ProgressCircle` in `src/App.tsx` | — |
| Vaul | On a phone the slot inspector is a sheet pulled from the bottom, not a full-screen dialog | `Drawer.Root` / `Drawer.Content` in `SlotDetail` | — |
| Sonner | A fill, an exit or a loop state change must announce itself once, stacked, without shifting the layout | `Toaster` + `toast` in `Console` | — |
| Motion | The "now" marker glides on each lane rail, candidate rows reorder, the window progress bar retargets | `motion.span layout`, `motion.li layout`, `motion.button` entrance, presets from `src/motion.ts` | — |

Search: the island filters the slot lanes and the tape log with the same query, and
opens a slot by its id (`onOpenItem` → the inspector).
