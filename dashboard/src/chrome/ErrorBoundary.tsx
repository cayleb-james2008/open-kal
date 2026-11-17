import { Component, type ErrorInfo, type ReactNode } from "react";

import { Button } from "@/components/ui/button";

type Props = { children: ReactNode };
type State = { error: Error | null; stack: string | null };

/**
 * A render error must read as a diagnosis, not a blank page. React unmounts the whole
 * tree when a component throws, which is how a renamed API field turned the dashboard
 * into a matte screen with no clue in it. This keeps the engine's state on screen and
 * prints the error, so the next breakage is visible instead of invisible.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, stack: null };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    this.setState({ stack: (info.componentStack ?? "").trim().slice(0, 1200) });
    console.error("[openkal] render error:", error);
  }

  render() {
    const { error, stack } = this.state;
    if (!error) return this.props.children;

    const detail = String(error.stack ?? `${error.name}: ${error.message}`).slice(0, 1600);
    return (
      <div className="min-h-dvh bg-canvas p-4 sm:p-6">
        <div
          data-openkal-error
          role="alert"
          className="mx-auto w-full max-w-2xl rounded-panel border border-hairline bg-surface p-5 shadow-card"
        >
          <div className="flex items-center gap-2">
            <span className="flex size-8 items-center justify-center rounded-lg bg-ink-button font-mono text-xs text-ink-button-fg">
              OK
            </span>
            <h1 className="text-ui font-medium">Open-Kal could not render this snapshot</h1>
          </div>
          <p className="mt-3 text-sm text-ink-2">
            The engine is unaffected and keeps running; only this page stopped. The error is below so it can be
            fixed rather than guessed at.
          </p>
          <pre className="mt-3 max-h-64 overflow-auto rounded-xl border border-hairline bg-board p-3 font-mono text-xs text-ink">
            {detail}
          </pre>
          {stack ? (
            <details className="mt-2 text-xs text-ink-2">
              <summary className="cursor-pointer">component stack</summary>
              <pre className="mt-2 max-h-48 overflow-auto rounded-xl border border-hairline bg-board p-3 font-mono text-xs">
                {stack}
              </pre>
            </details>
          ) : null}
          <div className="mt-3 flex items-center gap-3">
            <Button size="sm" onClick={() => window.location.reload()}>
              Reload
            </Button>
            <span className="font-mono text-xs text-ink-3">GET /api/state · {new Date().toISOString()}</span>
          </div>
        </div>
      </div>
    );
  }
}
