# LLM Prompt — A\* Grid Visualizer (Mouse & Cheese)

You are a senior full‑stack engineer. Build a **production‑quality, interactive A\* pathfinding visualizer** that demonstrates the algorithm step‑by‑step on a grid. The theme is **a mouse (start) finding cheese (target)**. Follow this prompt precisely.

---

## Objectives

* Educate users on how **A\*** works via a clear, animated, and controllable visualization.
* Provide **hands‑on interactivity**: adjustable grid, obstacles, start/target placement, speed control, step/run modes, heuristic selection, and cost display toggles.
* Ship **well‑structured, typed, tested** code with docs and easy local/dev deployment.

---

## Deliverables

1. **Source code** in a single repository with the structure defined below.
2. **Interactive desktop app** written in **Python** that runs locally with `pip install -r requirements.txt` and `python -m astar_viz`.
3. **Inline SVG** assets for mouse and cheese (vector). Mouse rotates to face travel direction.
4. **Unit tests** (**pytest**) with **table‑driven** cases covering valid, invalid, and edge scenarios.
5. **Well‑commented A\*** core with clear separation of concerns and explanatory inline docs.
6. **README.md** with screenshots/GIFs, feature overview, and usage instructions.

---

## Tech Stack & Tooling

* **GUI**: **PySide6 (Qt for Python)** with **QtSvg** for vector mouse/cheese rendering. Use **Qt Graphics View** (QGraphicsView/QGraphicsScene) for the grid and overlays.
* **Core Logic**: Pure Python A\* engine (framework‑agnostic) with a binary‑heap priority queue (`heapq`).
* **State/Architecture**: MVC-ish separation: `domain/` (pure logic) + `ui/` (Qt widgets) + `app/` (controllers, signals/slots). Use a small **finite‑state machine** for algorithm phases (Idle → Running → Paused → Complete → NoPath).
* **Styling**: Qt stylesheets with a light/dark/high‑contrast palette (suitable for big screens). Provide scalable fonts.
* **Testing**: **pytest** + **hypothesis** (optional) for property tests. Use **table‑driven tests** for algorithm core and controllers. **pytest‑qt** for basic UI interaction tests.
* **Quality**: **ruff** (lint), **black** (format), **mypy** (type checking, use `typing`/`pydantic` optional). **pre‑commit** hooks. **GitHub Actions** CI for lint + typecheck + tests.

---

## Core Concepts to Visualize

* **Grid** of nodes/tiles (default 25×25; adjustable 10–60 each dimension). Each node shows:

  * **G‑cost** (cost from start)
  * **H‑cost** (heuristic estimate to target)
  * **F‑cost = G + H**
* **Open set (frontier)**: nodes being considered (distinct color/border)
* **Closed set (visited)**: evaluated nodes (distinct color)
* **Walls/obstacles**: toggled by user interaction (click/drag)
* **Start node (mouse)** and **Target node (cheese)**: vector icons, colors must be distinct from tile colors
* **Path reconstruction**: animate the optimal path from target back to start using parent pointers

---

## Interaction & Controls (Top Toolbar)

Implement a **dockable Qt toolbar** (QToolBar) and **side panel** (QDockWidget) with the following controls:

* **Grid Size**: width/height spin boxes (10–60) + Apply (resets grid respectfully, preserving start/target when possible)
* **Random Map**: walls density slider (0–50%) + seed field for reproducibility
* **Random Start/Target**: button; ensure no overlap and not on walls
* **Edit Modes** (QButtonGroup toggle): **Add Walls**, **Erase Walls**, **Set Start**, **Set Target** (grid supports click‑drag painting for walls)
* **Run Controls**: `Step`, `Run`, `Pause`, `Reset`
* **Speed**: slider (50–1000 ms) controlling timer interval
* **Cost Display**: checkboxes **G / H / F** and master switch “Show All Costs”
* **Heuristic**: combo box — **Manhattan**, **Euclidean**, **Diagonal (Chebyshev)**, **Octile** with tooltip help
* **Movement**: radio buttons **4‑dir** or **8‑dir**; checkbox **Disallow corner‑cutting**
* **Weights (optional)**: toggle to enable weighted tiles; brush to paint weight (>1); legend
* **Theme**: light/dark/high‑contrast theme combo

---

## Algorithm Requirements (A\* Implementation)

* Priority queue (binary heap) for open set keyed by **F‑cost**, with tie‑break on lower **H** then lower **G**.
* Compute:

  * `g(n)` accumulated from start (1 for orthogonal move; √2 for diagonal unless weights override)
  * `h(n)` based on selected heuristic
  * `f(n) = g(n) + h(n)`
* Store `parent` for each visited node to reconstruct path.
* **Step Mode**: one expansion per click; highlight current node; animate neighbor relaxations.
* **Run Mode**: iterate under `requestAnimationFrame` or setInterval respecting speed control; allow `Pause`.
* **Completion States**: `FOUND` (animate path tracing), `NO_PATH` (flash a message and highlight sealed regions), `CANCELLED` (on Reset).
* **Determinism**: When costs tie, prefer a stable ordering to keep visuals reproducible.

---

## Visual & UX Details

* **Tile rendering**: Draw via QGraphicsItem with smooth hover hints; subtle transitions; rounded rects. Ensure **distinct colors**:

  * Empty tile
  * Wall
  * Open set (frontier)
  * Closed set (visited)
  * Current node (pulse animation via QPropertyAnimation)
  * Final path (glow effect via drop shadow/outer glow)
* **Costs layout**: small monospaced labels within each tile: top‑left **G**, top‑right **H**, bottom‑center **F**. Hide gracefully when toggled off or tile too small.
* **Mouse & Cheese (SVG)**:

  * Render with **QtSvg**; the **mouse head rotates** to face the last movement direction along the current reconstructed path (or target direction when idle). Keep mouse color distinct from tiles; cheese color distinct.
  * Scale cleanly on high‑DPI displays.
* **Accessibility**: keyboard shortcuts for major actions; accessible names for controls; color‑blind‑safe palette; persistent tooltips.
* **Big Screen Ready**: base font 16–18px, scalable; toolbar buttons large enough (min 40×40 px hit area); responsive window layout.

---

## Architecture & File Layout

```
/ (repo root)
  ├─ astar_viz/
  │   ├─ app/
  │   │   ├─ __init__.py
  │   │   ├─ controller.py             # glue between UI and domain; FSM for run modes
  │   │   └─ fsm.py                    # Idle/Running/Paused/Complete/NoPath
  │   ├─ ui/
  │   │   ├─ main_window.py            # QMainWindow + toolbars/docks
  │   │   ├─ grid_view.py              # QGraphicsView/QGraphicsScene grid renderer
  │   │   ├─ tiles.py                  # QGraphicsItem nodes: draw states & costs
  │   │   ├─ svg_mouse.py              # rotatable SVG mouse (QtSvg + transforms)
  │   │   └─ svg_cheese.py             # SVG cheese icon
  │   ├─ domain/
  │   │   ├─ __init__.py
  │   │   ├─ types.py                  # dataclasses/TypedDicts for Node, Grid, Costs, Heuristic
  │   │   ├─ heuristics.py             # manhattan/euclidean/diagonal/octile
  │   │   ├─ neighbors.py              # 4/8 dirs + corner‑cut logic
  │   │   ├─ priority_queue.py         # thin wrapper over heapq with tie‑breaks
  │   │   ├─ astar.py                  # pure A* core (no Qt)
  │   │   └─ path.py                   # reconstruct path; direction utilities
  │   ├─ utils/
  │   │   ├─ grid_factory.py           # create/reset/randomize grids, weights
  │   │   └─ rng.py                    # seeded RNG
  │   ├─ assets/
  │   │   ├─ mouse.svg
  │   │   └─ cheese.svg
  │   └─ __main__.py                   # entry point: `python -m astar_viz`
  ├─ tests/
  │   ├─ test_astar.py                 # table‑driven A* tests
  │   ├─ test_heuristics.py            # admissibility/consistency checks
  │   ├─ test_neighbors.py             # movement rules & corner‑cutting
  │   ├─ test_controller.py            # step/run/pause/reset behaviors (timer mocked)
  │   └─ test_ui_smoke.py              # pytest‑qt smoke: window opens; basic actions
  ├─ README.md
  ├─ requirements.txt
  ├─ pyproject.toml                    # ruff/black/mypy config (optional)
  ├─ .pre-commit-config.yaml
  └─ .github/workflows/ci.yml          # run lint/type/test on push/PR
```

---

## Type Definitions (Guidance)

Use **Python dataclasses and type hints** instead of TypeScript interfaces. Example:

```python
# astar_viz/domain/types.py
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

Coord = Tuple[int, int]

NodeState = Literal[
    "empty", "wall", "start", "target",
    "open", "closed", "path", "current"
]

@dataclass
class Costs:
    g: float
    h: float
    f: float

@dataclass
class GridNode:
    id: str
    coord: Coord
    cost: Costs
    walkable: bool = True
    weight: float = 1.0
    state: NodeState = "empty"
    parent: Optional[str] = None

@dataclass
class Grid:
    width: int
    height: int
    nodes: dict[str, GridNode]

HeuristicId = Literal["manhattan", "euclidean", "diagonal", "octile"]

@dataclass
class AlgoConfig:
    diag: bool
    corner_cutting: bool
    heuristic: HeuristicId
```

---

## Testing Requirements

**General**

* Use **pytest** with **table‑driven tests** (parametrized with `@pytest.mark.parametrize`) to validate algorithm correctness and controller behaviors.
* Cover **valid, invalid, and edge cases**. Include property‑based tests with **hypothesis** (optional) for path optimality vs Dijkstra on small random grids.

**A\* Core (test\_astar.py)**

* Cases: small empty grids (2×2, 3×3), larger grids (25×25), grids with no path, grids with heavy weights, start==target, start/target on borders, dense walls (30–50%).
* Verify: returned path optimality (sum of weights equals minimal), expansions count monotonicity with tighter heuristics, tie‑break determinism.
* Invalid inputs: negative sizes, out‑of‑bounds start/target, NaN weights → expect `ValueError` with helpful messages.

**Heuristics (test\_heuristics.py)**

* Check that `h(n)` ≤ true shortest distance for admissible heuristics under the chosen movement rules; include a **counter‑case** when rules change (e.g., diagonals disabled) to ensure tests adapt.

**Neighbors (test\_neighbors.py)**

* Validate 4‑ vs 8‑connected neighborhoods and corner‑cutting rules (disallow diagonal when both adjacent orthogonals are walls).

**Controller/UI (test\_controller.py, test\_ui\_smoke.py)**

* Controller state transitions (Step/Run/Pause/Reset) with a mocked QTimer.
* Speed slider alters tick cadence.
* Cost toggles hide/show labels; heuristic dropdown changes `h()` used by core.

**Coverage Targets**

* ≥ 90% for `domain/` (astar, heuristics, neighbors, priority\_queue)
* ≥ 80% overall

---

## Implementation Notes & Best Practices

* Keep **A\*** **framework‑agnostic** in `domain/astar.ts` (pure functions). UI calls it via a small adapter.
* Use a **binary heap** PQ; avoid O(n) scans.
* Maintain **immutable** updates to store slices; batch updates when stepping to minimize re‑renders.
* Use `requestAnimationFrame` for run animation; respect speed slider by skipping frames based on computed interval.
* All colors and layout tokens via Tailwind config; ensure color‑blind‑friendly contrast.
* Guard invalid states; provide toasts for user errors (e.g., trying to run without start/target).
* Include **seeded RNG** for reproducible demos during talks.
* Export a **`createPreset(density, seed)`** helper to quickly generate demo maps.

---

## Documentation (README.md)

* Overview, features, screenshots/GIF.
* A\* primer with mini diagrams (markdown + inline SVG).
* How to run, test, build, and configure.
* Keyboard shortcuts reference.
* Notes on heuristics and their impact.

---

## Acceptance Criteria

1. User can place/remove walls with drag; place start/target with tools.
2. Open/Closed/Current nodes clearly distinguished in color and legend.
3. G/H/F costs render per tile and can be toggled.
4. Step/Run/Pause/Reset work; speed control affects animation rate.
5. Heuristic and movement options are switchable at runtime.
6. Path reconstruction animates with a glow; mouse rotation reflects travel direction.
7. Works smoothly up to 60×60 grid on a modern laptop; no frame jank at default speed.
8. Tests pass locally and in CI; coverage thresholds met.

---

## Output Format Instructions

* Provide the **complete repository** in the answer using well‑labeled code blocks per file path.
* Include **all configs** (`requirements.txt`, optional `pyproject.toml`, CI).
* Ensure the project runs with:

  * `python -m venv .venv && source .venv/bin/activate` (or Windows equivalent)
  * `pip install -r requirements.txt`
  * `python -m astar_viz`

---

## Optional Nice‑to‑Haves (Only after core passes tests)

* Weighted brush to paint terrain costs (sand, mud) and show a small legend.
* Export/Import grid as JSON.
* Snapshot/Replay of algorithm steps.
* Playwright e2e smoke test for “grid 10×10, run A\*, path exists”.

---

## License & Attribution

* Include a permissive license (MIT) at repo root.
* Add short attribution in README: “Mouse & Cheese A\* Visualizer inspired by classic pathfinding demos.”

---

## Open Questions for the Requester (answer briefly; defaults below will be used otherwise)

1. **Default grid size** (default **25×25**)?
2. **Diagonal movement & corner‑cutting** defaults (default **8‑dir, no corner‑cutting**)?
3. **Max grid** (default **60×60**)?
4. **Color theme** preferences or brand colors (default **neutral/high‑contrast palette**)?
5. **Include optional weighted tiles** (default **excluded** in v1)?

---

**Proceed to generate the codebase exactly per the above.** Ensure best practices, clean architecture, strong typing, and comprehensive unit tests with table‑driven cases.
