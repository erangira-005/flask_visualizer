"""
Time Complexity Visualizer - Flask Server
Endpoint: GET /analyze?algo=linear_search&step=10&n_max=10000
"""

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import os
import base64
import time
import math
import json
import traceback
from io import BytesIO
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  Database (SQLAlchemy)
# ─────────────────────────────────────────────
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + \
    os.path.join(BASE_DIR, "analyses.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


class Analysis(db.Model):
    """One row per /analyze run."""
    __tablename__ = "analyses"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(
        db.DateTime, default=datetime.utcnow, nullable=False)
    method = db.Column(db.String(4), nullable=False)          # "GET" or "POST"
    step = db.Column(db.Integer, nullable=False)
    n_min = db.Column(db.Integer, nullable=False, default=0)
    n_max = db.Column(db.Integer, nullable=False)
    elapsed_ms = db.Column(db.Float, nullable=False)
    snapshot_path = db.Column(db.String(512), nullable=False)
    image_base64 = db.Column(db.Text, nullable=False)

    results = db.relationship(
        "AlgorithmResult",
        backref="analysis",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def to_dict(self, include_image: bool = False) -> dict:
        data = {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "method": self.method,
            "parameters": {
                "algorithms": [r.name for r in self.results],
                "step": self.step,
                "n_min": self.n_min,
                "n_max": self.n_max,
            },
            "algorithms": {r.name: r.to_dict() for r in self.results},
            "snapshot_path": self.snapshot_path,
            "elapsed_ms": self.elapsed_ms,
        }
        if include_image:
            data["image_base64"] = self.image_base64
        return data


class AlgorithmResult(db.Model):
    """One row per algorithm per analysis run."""
    __tablename__ = "algorithm_results"

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(
        db.Integer, db.ForeignKey("analyses.id"), nullable=False, index=True)
    name = db.Column(db.String(64), nullable=False)
    label = db.Column(db.String(128), nullable=False)
    complexity = db.Column(db.String(32), nullable=False)
    ops_at_n_max = db.Column(db.Float, nullable=True)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "complexity": self.complexity,
            "ops_at_n_max": self.ops_at_n_max,
        }


with app.app_context():
    db.create_all()


def save_analysis(method, requested, step, n_max, algo_data,
                  filepath, elapsed_ms, img_b64) -> Analysis:
    """Persist one analysis run (and its per-algorithm results) to the DB."""
    analysis = Analysis(
        method=method,
        step=step,
        n_min=0,
        n_max=n_max,
        elapsed_ms=elapsed_ms,
        snapshot_path=filepath,
        image_base64=img_b64,
    )
    for name in requested:
        info = algo_data[name]
        analysis.results.append(AlgorithmResult(
            name=name,
            label=info["label"],
            complexity=info["complexity"],
            ops_at_n_max=info["ops_at_n_max"],
        ))

    try:
        db.session.add(analysis)
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        raise
    return analysis


# ─────────────────────────────────────────────
#  Algorithm definitions
#  Each entry: label, complexity_label, fn(n)->ops
# ─────────────────────────────────────────────
ALGORITHMS = {
    # O(1) – constant
    "constant": {
        "label": "Constant Access",
        "complexity": "O(1)",
        "color": "#27ae60",
        "ops_fn": lambda n: 1,
    },
    # O(log n) – binary search
    "binary_search": {
        "label": "Binary Search",
        "complexity": "O(log n)",
        "color": "#2980b9",
        "ops_fn": lambda n: math.log2(n) if n > 0 else 0,
    },
    # O(n) – linear search
    "linear_search": {
        "label": "Linear Search",
        "complexity": "O(n)",
        "color": "#8e44ad",
        "ops_fn": lambda n: n,
    },
    # O(n log n) – merge sort / heap sort
    "merge_sort": {
        "label": "Merge Sort",
        "complexity": "O(n log n)",
        "color": "#d35400",
        "ops_fn": lambda n: n * math.log2(n) if n > 1 else n,
    },
    # O(n²) – bubble sort
    "bubble_sort": {
        "label": "Bubble Sort",
        "complexity": "O(n²)",
        "color": "#c0392b",
        "ops_fn": lambda n: n ** 2,
    },
    # O(n²) – insertion sort
    "insertion_sort": {
        "label": "Insertion Sort",
        "complexity": "O(n²)",
        "color": "#e67e22",
        "ops_fn": lambda n: n ** 2,
    },
    # O(n³) – nested loops
    "nested_loops": {
        "label": "Nested Loops (3×)",
        "complexity": "O(n³)",
        "color": "#7f8c8d",
        "ops_fn": lambda n: n ** 3,
    },
    # O(2ⁿ) – exponential (exponent capped so large n can't overflow a float)
    "exponential": {
        "label": "Exponential",
        "complexity": "O(2ⁿ)",
        "color": "#e74c3c",
        "ops_fn": lambda n: min(2.0 ** min(n, 60), 1e15),
    },
    # O(n!) – factorial
    "factorial": {
        "label": "Factorial",
        "complexity": "O(n!)",
        "color": "#c0392b",
        "ops_fn": lambda n: min(math.factorial(int(n)) if n <= 20 else float('inf'), 1e15),
    },
    # O(sqrt n)
    "sqrt_n": {
        "label": "Square Root",
        "complexity": "O(√n)",
        "color": "#16a085",
        "ops_fn": lambda n: math.sqrt(n),
    },

    # ── STACK algorithms ──────────────────────────────────────
    "stack_push_pop": {
        "label": "Stack Push/Pop",
        "complexity": "O(1)",
        "color": "#27ae60",
        "ops_fn": lambda n: 1,
    },
    "stack_reverse": {
        "label": "Stack Reverse",
        "complexity": "O(n)",
        "color": "#9b59b6",
        "ops_fn": lambda n: n,
    },
    "stack_sort": {
        "label": "Stack Sort",
        "complexity": "O(n²)",
        "color": "#e67e22",
        "ops_fn": lambda n: n ** 2,
    },
    "stack_balanced_brackets": {
        "label": "Balanced Brackets",
        "complexity": "O(n)",
        "color": "#1abc9c",
        "ops_fn": lambda n: n,
    },

    # ── QUEUE algorithms ──────────────────────────────────────
    "queue_enqueue_dequeue": {
        "label": "Queue Enqueue/Dequeue",
        "complexity": "O(1)",
        "color": "#2ecc71",
        "ops_fn": lambda n: 1,
    },
    "queue_bfs": {
        "label": "Queue BFS",
        "complexity": "O(n + e)",
        "color": "#3498db",
        "ops_fn": lambda n: n,
    },
    "queue_hot_potato": {
        "label": "Hot Potato",
        "complexity": "O(n × k)",
        "color": "#e74c3c",
        "ops_fn": lambda n: n * math.log2(n) if n > 1 else n,
    },
}

VALID_ALGOS = set(ALGORITHMS.keys())


def parse_algos(raw: str) -> list[str]:
    """Accept comma-separated or JSON-array algo names."""
    raw = raw.strip()
    if raw.startswith("["):
        try:
            names = json.loads(raw)
            if isinstance(names, list):
                return [n.strip().strip("'\"") for n in names]
        except json.JSONDecodeError:
            pass
    return [a.strip().strip("'\"") for a in raw.split(",")]


def build_chart(algos: list[str], step: int, n_max: int) -> tuple[plt.Figure, np.ndarray]:
    """Return (fig, x_values) with the complexity curves plotted."""
    x = np.arange(step, n_max + 1, step, dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    for name in algos:
        info = ALGORITHMS[name]
        y = np.array([info["ops_fn"](xi) for xi in x], dtype=float)
        ax.plot(x, y, label=f"{info['label']} {info['complexity']}",
                color=info["color"], linewidth=2.2, alpha=0.92)

    ax.set_xlabel("Input size  n", color="#cdd6f4", fontsize=13)
    ax.set_ylabel("Operations (approx.)", color="#cdd6f4", fontsize=13)
    ax.set_title("Time Complexity Visualizer", color="#cdd6f4",
                 fontsize=16, fontweight="bold", pad=14)
    ax.tick_params(colors="#a6adc8", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#313244")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}" if v < 1e6 else f"{v:.2e}"
    ))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(color="#313244", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(facecolor="#1e2030", edgecolor="#45475a",
              labelcolor="#cdd6f4", fontsize=10, loc="upper left")
    fig.tight_layout()
    return fig, x


# ─────────────────────────────────────────────
#  /analyze endpoint (GET)
# ─────────────────────────────────────────────
@app.route("/analyze", methods=["GET"])
def analyze():
    algo_raw = request.args.get("algo", "")
    step_raw = request.args.get("step", "10")
    n_max_raw = request.args.get("n_max", "1000")

    # algo
    if not algo_raw:
        return jsonify({"error": "Missing required parameter: algo",
                        "valid_algorithms": sorted(VALID_ALGOS)}), 400
    requested = parse_algos(algo_raw)
    unknown = [a for a in requested if a not in VALID_ALGOS]
    if unknown:
        return jsonify({"error": f"Unknown algorithm(s): {unknown}",
                        "valid_algorithms": sorted(VALID_ALGOS)}), 400

    # step
    try:
        step = int(step_raw.replace(",", ""))
        if step < 1:
            raise ValueError
    except ValueError:
        return jsonify({"error": f"Invalid step value: {step_raw!r}. Must be a positive integer."}), 400

    # n_max
    try:
        n_max = int(n_max_raw.replace(",", ""))
        if n_max < step:
            raise ValueError("n_max must be >= step")
    except ValueError as exc:
        return jsonify({"error": f"Invalid n_max value: {n_max_raw!r}. {exc}"}), 400

    # generate chart
    t0 = time.perf_counter()
    try:
        fig, x_vals = build_chart(requested, step, n_max)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Chart generation failed: {exc}"}), 500

    # save snapshot
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"complexity_{ts}.png"
    filepath = os.path.join(SNAPSHOTS_DIR, filename)
    fig.savefig(filepath, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())

    # encode to base64
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    # build per-algo data
    algo_data = {}
    x_list = list(x_vals)
    for name in requested:
        info = ALGORITHMS[name]
        ops = [info["ops_fn"](xi) for xi in x_list]
        algo_data[name] = {
            "label": info["label"],
            "complexity": info["complexity"],
            "ops_at_n_max": round(ops[-1], 4) if ops else None,
        }

    # save to database
    try:
        analysis = save_analysis("GET", requested, step, n_max, algo_data,
                                 filepath, elapsed_ms, img_b64)
    except SQLAlchemyError as exc:
        traceback.print_exc()
        return jsonify({"error": f"Database save failed: {exc}"}), 500

    response = {
        "status": "ok",
        "analysis_id": analysis.id,
        "parameters": {
            "algorithms": requested,
            "step": step,
            "n_min": 0,
            "n_max": n_max,
        },
        "algorithms": algo_data,
        "snapshot_path": filepath,
        "elapsed_ms": elapsed_ms,
        "image_base64": img_b64,
    }
    return jsonify(response), 200


# ─────────────────────────────────────────────
#  /analyze endpoint (POST)
# ─────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze_post():
    # get the JSON body from the request
    body = request.get_json(silent=True)

    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    # extract parameters from the body instead of the URL
    algo_raw = body.get("algo", "")
    step_raw = str(body.get("step", "10"))
    n_max_raw = str(body.get("n_max", "1000"))

    # algo
    if not algo_raw:
        return jsonify({"error": "Missing required parameter: algo",
                        "valid_algorithms": sorted(VALID_ALGOS)}), 400

    # handle both string and list formats
    if isinstance(algo_raw, list):
        requested = algo_raw
    else:
        requested = parse_algos(algo_raw)

    unknown = [a for a in requested if a not in VALID_ALGOS]
    if unknown:
        return jsonify({"error": f"Unknown algorithm(s): {unknown}",
                        "valid_algorithms": sorted(VALID_ALGOS)}), 400

    # step
    try:
        step = int(str(step_raw).replace(",", ""))
        if step < 1:
            raise ValueError
    except ValueError:
        return jsonify({"error": f"Invalid step value: {step_raw}"}), 400

    # n_max
    try:
        n_max = int(str(n_max_raw).replace(",", ""))
        if n_max < step:
            raise ValueError("n_max must be >= step")
    except ValueError as exc:
        return jsonify({"error": f"Invalid n_max value: {n_max_raw}. {exc}"}), 400

    # generate chart
    t0 = time.perf_counter()
    try:
        fig, x_vals = build_chart(requested, step, n_max)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Chart generation failed: {exc}"}), 500

    # save snapshot
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filepath = os.path.join(SNAPSHOTS_DIR, f"complexity_{ts}.png")
    fig.savefig(filepath, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())

    # encode to base64
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    algo_data = {}
    for name in requested:
        info = ALGORITHMS[name]
        ops = [info["ops_fn"](xi) for xi in x_vals]
        algo_data[name] = {
            "label": info["label"],
            "complexity": info["complexity"],
            "ops_at_n_max": round(ops[-1], 4) if ops else None,
        }

    # save to database
    try:
        analysis = save_analysis("POST", requested, step, n_max, algo_data,
                                 filepath, elapsed_ms, img_b64)
    except SQLAlchemyError as exc:
        traceback.print_exc()
        return jsonify({"error": f"Database save failed: {exc}"}), 500

    return jsonify({
        "status": "ok",
        "method": "POST",
        "analysis_id": analysis.id,
        "parameters": {
            "algorithms": requested,
            "step": step,
            "n_min": 0,
            "n_max": n_max,
        },
        "algorithms": algo_data,
        "snapshot_path": filepath,
        "elapsed_ms": elapsed_ms,
        "image_base64": img_b64,
    }), 200


# ─────────────────────────────────────────────
#  /analyses – read saved analyses from the DB
# ─────────────────────────────────────────────
@app.route("/analyses", methods=["GET"])
def list_analyses():
    try:
        limit = min(int(request.args.get("limit", 20)), 100)
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400

    rows = (Analysis.query
            .order_by(Analysis.created_at.desc())
            .limit(limit)
            .all())
    return jsonify({"count": len(rows),
                    "analyses": [a.to_dict() for a in rows]})


@app.route("/analyses/<int:analysis_id>", methods=["GET"])
def get_analysis(analysis_id):
    analysis = db.session.get(Analysis, analysis_id)
    if analysis is None:
        return jsonify({"error": f"Analysis {analysis_id} not found"}), 404
    return jsonify(analysis.to_dict(include_image=True))


# ─────────────────────────────────────────────
#  /algorithms – discovery endpoint
# ─────────────────────────────────────────────
@app.route("/algorithms", methods=["GET"])
def list_algorithms():
    return jsonify({
        "algorithms": {
            name: {"label": info["label"], "complexity": info["complexity"]}
            for name, info in ALGORITHMS.items()
        }
    })


# ─────────────────────────────────────────────
#  /health
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "snapshot_dir": SNAPSHOTS_DIR})


if __name__ == "__main__":
    print("=" * 60)
    print("  Time Complexity Visualizer  –  http://localhost:8000")
    print("=" * 60)
    print(f"  Supported algorithms: {', '.join(sorted(ALGORITHMS.keys()))}")
    print(f"  Snapshots saved to:   {SNAPSHOTS_DIR}")
    print()
    print("  Example:")
    print("  http://localhost:8000/analyze?algo=linear_search,bubble_sort&step=10&n_max=1000")
    print()
    app.run(host="0.0.0.0", port=8000, debug=False)
