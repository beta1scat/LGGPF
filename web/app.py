"""Flask web application for the LGGPF grasping pipeline.

Provides a REST API that exposes each stage of the pipeline
(load models, capture image, detect, segment, fit, plan, execute)
as a separate HTTP endpoint.  The frontend is served from
``web/templates/index.html``.

Usage::

    cd lggpf
    python -m flask --app web.app run --port 5000

Or programmatically::

    from web.app import create_app
    app = create_app()
    app.run(debug=False)
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, render_template, request, jsonify

from lggpf.pipeline import GraspingPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

_pipeline: GraspingPipeline | None = None


def get_pipeline() -> GraspingPipeline:
    """Return the singleton pipeline instance, creating it if needed."""
    global _pipeline
    if _pipeline is None:
        _pipeline = GraspingPipeline()
    return _pipeline


def create_app(config_path: str | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config_path: Optional path to a YAML config file.  If *None*,
            the default ``config/default.yaml`` is used.

    Returns:
        Configured Flask app.
    """
    global _pipeline

    template_dir = str(Path(__file__).resolve().parent / "templates")
    app = Flask(__name__, template_folder=template_dir)

    # Initialize pipeline
    if config_path is not None:
        from lggpf.config import load_config

        cfg = load_config(config_path)
        _pipeline = GraspingPipeline(cfg)
    else:
        _pipeline = GraspingPipeline()

    # Register routes
    _register_routes(app)
    return app


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def _register_routes(app: Flask) -> None:
    """Register all REST endpoints on the Flask app."""

    @app.route("/")
    def index():
        """Render the main page and initialize the robot model."""
        pipeline = get_pipeline()
        if "robot" not in pipeline.models:
            pipeline.init_robot_only()
        return render_template("index.html")

    @app.route("/load_models", methods=["POST"])
    def load_models():
        """Load all required models and connect to hardware."""
        try:
            pipeline = get_pipeline()
            pipeline.load_models()
            return jsonify(
                {"status": "success", "message": "Models loaded successfully."}
            )
        except Exception as e:
            logger.exception("Failed to load models.")
            return jsonify({"status": "error", "message": str(e)})

    @app.route("/capture_image", methods=["POST"])
    def capture_image():
        """Capture RGB and depth images from the camera."""
        try:
            pipeline = get_pipeline()
            img_base64, time_str = pipeline.capture_image()
            return jsonify(
                {
                    "status": "success",
                    "time_str": time_str,
                    "image": img_base64,
                }
            )
        except Exception as e:
            logger.exception("Failed to capture image.")
            return jsonify({"status": "error", "message": str(e)})

    @app.route("/get_boxes", methods=["POST"])
    def get_boxes():
        """Detect bounding boxes from user text query."""
        try:
            pipeline = get_pipeline()
            text = request.json.get("text", "")
            num_boxes = pipeline.detect_objects(text)
            return jsonify({"status": "success", "boxes": num_boxes})
        except ValueError as e:
            return jsonify({"status": "failed", "message": str(e)})
        except Exception as e:
            logger.exception("Failed to detect objects.")
            return jsonify({"status": "error", "message": str(e)})

    @app.route("/view_box", methods=["POST"])
    def view_box():
        """Select and display a specific bounding box."""
        try:
            pipeline = get_pipeline()
            box_index = request.json.get("box_index", 1)
            box, img_base64 = pipeline.select_box(box_index)
            return jsonify(
                {
                    "status": "success",
                    "box": box,
                    "image": img_base64,
                }
            )
        except Exception as e:
            logger.exception("Failed to view box.")
            return jsonify({"status": "error", "message": str(e)})

    @app.route("/segment_object", methods=["POST"])
    def segment_object():
        """Segment the object within the selected bounding box."""
        try:
            pipeline = get_pipeline()
            point_count = pipeline.segment_object()
            return jsonify({"status": "success", "point_count": point_count})
        except Exception as e:
            logger.exception("Failed to segment object.")
            return jsonify({"status": "error", "message": str(e)})

    @app.route("/class_and_fit", methods=["POST"])
    def class_and_fit():
        """Classify and fit a geometric primitive to the point cloud."""
        try:
            pipeline = get_pipeline()
            result = pipeline.classify_and_fit()
            return jsonify(
                {
                    "status": "success",
                    "params": result["params"],
                    "category": result["category"],
                }
            )
        except Exception as e:
            logger.exception("Failed to classify and fit.")
            return jsonify({"status": "error", "message": str(e)})

    @app.route("/generate_pick_poses", methods=["POST"])
    def generate_pick_poses():
        """Generate candidate grasp poses for the fitted shape."""
        try:
            pipeline = get_pipeline()
            num_poses = pipeline.generate_pick_poses()
            return jsonify({"status": "success", "pick_poses": num_poses})
        except Exception as e:
            logger.exception("Failed to generate pick poses.")
            return jsonify({"status": "error", "message": str(e)})

    @app.route("/plan_robot", methods=["POST"])
    def plan_robot():
        """Plan a collision-free trajectory to the selected grasp pose."""
        try:
            pipeline = get_pipeline()
            pick_pose_index = request.json.get("pick_pose_index", 1)
            success = pipeline.plan_trajectory(pick_pose_index)
            if success:
                return jsonify(
                    {
                        "status": "success",
                        "message": "Robot planned successfully.",
                    }
                )
            else:
                return jsonify(
                    {
                        "status": "failed",
                        "message": "Planning failed (IK or collision).",
                    }
                )
        except Exception as e:
            logger.exception("Failed to plan trajectory.")
            return jsonify({"status": "error", "message": str(e)})

    @app.route("/exec_robot", methods=["POST"])
    def exec_robot():
        """Execute the planned trajectory on the robot."""
        try:
            pipeline = get_pipeline()
            pipeline.execute()
            return jsonify(
                {
                    "status": "success",
                    "message": "Robot executed successfully.",
                }
            )
        except Exception as e:
            logger.exception("Failed to execute trajectory.")
            return jsonify({"status": "error", "message": str(e)})


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.run(debug=False)
