"""``preview``: derived stills and a three-view video from a run capture.

Previews are diagnostics. They never enter the official dataset and never
replace the per-camera recordings.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from soft_sim.recording.reader import capture_directory, first_existing, run_directory
from soft_sim.storage import Storage, json_default

#: Panel sizes of the three-view composite, in pixels.
FINGER_PANEL = 240
EXTERNAL_PANEL = (640, 480)
CJK_FONTS = (
    "/home/three/.local/share/fonts/PingFang Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)
VIEW_LABELS = {"left_rgb": "左指内相机", "right_rgb": "右指内相机", "external_rgb": "外部相机"}


def montage(frames, path):
    """Tile the three camera views of one recorded frame into one PNG."""
    from PIL import Image
    panels = []
    for name in ("left_rgb", "right_rgb", "external_rgb"):
        panel = np.asarray(frames[name], dtype=np.uint8)
        if panel.shape[:2] != (240, 320):
            panel = np.asarray(Image.fromarray(panel).resize((320, 240)))
        panels.append(panel)
    Image.fromarray(np.concatenate(panels, axis=1)).save(path)


def build(run, output=None, count=4):
    from soft_sim.recording.reader import open_frame
    capture = capture_directory(run)
    episode = json.loads((capture / "episode.json").read_text())
    rows = episode["frames"]
    if not rows:
        raise ValueError("Capture has no frames to preview")
    preview_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(output) if output else capture.parent / "previews" / preview_id
    output.mkdir(parents=True, exist_ok=False)
    indices = sorted(set(np.linspace(0, len(rows) - 1, min(count, len(rows))).astype(int)))
    written = []
    for index in indices:
        with open_frame(capture, rows[index]) as frame:
            path = output / f"frame_{index:06d}.png"
            montage(frame, path)
            written.append(dict(index=index, time_s=rows[index]["time_s"],
                                phase=rows[index]["phase"], file=path.name))
    (output / "preview.json").write_text(json.dumps(dict(
        preview_id=preview_id, capture=str(capture), frame_count=len(rows),
        frames=written, layout="left | right | external"), indent=2, default=json_default))
    return output, written


# -- three-view video -----------------------------------------------------
def _font(size):
    from PIL import ImageFont
    for candidate in CJK_FONTS:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _fit(image, height, width=None):
    """Resize keeping the aspect ratio, padding to the requested panel size."""
    from PIL import Image
    image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if width is None:
        width = round(image.width * height / image.height)
    return np.asarray(image.resize((width, height), Image.LANCZOS))


def compose(left, right, external):
    """External camera on the left, the two finger cameras stacked on the right."""
    height = EXTERNAL_PANEL[1]
    fingers = [np.ascontiguousarray(_fit(panel, FINGER_PANEL, FINGER_PANEL))
               for panel in (left, right)]
    external = np.ascontiguousarray(_fit(external, height, EXTERNAL_PANEL[0]))
    return np.concatenate([external, np.concatenate(fingers, axis=0)], axis=1)


def annotate(image, labels, footer):
    """Draw panel labels and one footer line; returns a new RGB array."""
    from PIL import Image, ImageDraw
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    title, small = _font(20), _font(16)
    for text, xy in labels:
        box = draw.textbbox(xy, text, font=title)
        draw.rectangle((box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2), fill=(0, 0, 0))
        draw.text(xy, text, font=title, fill=(255, 255, 255))
    if footer:
        x, y = 10, canvas.height - 26
        box = draw.textbbox((x, y), footer, font=small)
        draw.rectangle((box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2), fill=(0, 0, 0))
        draw.text((x, y), footer, font=small, fill=(235, 235, 235))
    return np.asarray(canvas)


class VideoWriter:
    """Prefer H.264 through PyAV; fall back to OpenCV's mp4v encoder."""

    def __init__(self, path, fps, size):
        self.path, self.fps, self.size = Path(path), float(fps), size
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = None
        try:
            import av
            self.container = av.open(str(self.path), mode="w")
            self.stream = self.container.add_stream("libx264", rate=int(round(fps)))
            self.stream.width, self.stream.height = size
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"crf": "20"}
            self.backend = "pyav"
        except Exception:
            import cv2
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                          float(fps), size)
            if not self.writer.isOpened():
                raise RuntimeError(f"Cannot create video: {self.path}")
            self.backend = "cv2"

    def write(self, image):
        if self.backend == "pyav":
            import av
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
        else:
            import cv2
            self.writer.write(cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR))

    def close(self):
        if self.backend == "pyav":
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
        else:
            self.writer.release()


def _task_summary(run_dir, material=None):
    result = {}
    try:
        result = json.loads(first_existing(run_dir / "results/task.json",
                                           run_dir / "result.json").read_text())
    except FileNotFoundError:
        pass
    verdict = {True: "成功", False: "未成功"}.get(result.get("success"), "无任务结论")
    task = result.get("task") or "unknown task"
    material = material or result.get("config", {}).get("material", {})
    material_name = material.get("name", "").split(" (")[0] or "软指"
    color_name = material.get("color_name", "").split(" / ")[0]
    if "custom appearance" in material.get("color_name", ""):
        color_name += "（自定义）"
    variant = f"原双 X5 + {material_name} {color_name}" if result.get("official_task_manager") else ""
    return " · ".join(part for part in (task, variant, f"RewardManager {verdict}") if part)


def build_video(run, output, *, fps=None, limit=None):
    from soft_sim.recording.reader import open_frame
    capture = capture_directory(run)
    run_dir = run_directory(capture)
    episode = json.loads((capture / "episode.json").read_text())
    rows = episode["frames"][:limit] if limit else episode["frames"]
    if not rows:
        raise ValueError("Capture has no frames")
    fps = float(fps or episode["fps"])
    calibration = episode.get("calibration", {})
    summary = _task_summary(run_dir, calibration.get("material"))
    phases = list(episode.get("phases") or [])
    external_label = "桌面相机" if calibration.get("external_camera_name") == "cam_head" else "外部相机"
    labels = [(external_label, (8, 6)), ("左指内相机", (EXTERNAL_PANEL[0] + 8, 6)),
              ("右指内相机", (EXTERNAL_PANEL[0] + 8, FINGER_PANEL + 6))]
    size = (EXTERNAL_PANEL[0] + FINGER_PANEL, EXTERNAL_PANEL[1])
    writer = VideoWriter(output, fps, size)
    try:
        for index, row in enumerate(rows):
            with open_frame(capture, row) as frame:
                image = compose(frame["left_rgb"], frame["right_rgb"], frame["external_rgb"])
            phase = row.get("phase")
            if isinstance(phase, (int, np.integer)) and 0 <= int(phase) < len(phases):
                phase = phases[int(phase)]
            footer = f"{summary} · {row['time_s']:.2f}s · {phase}"
            writer.write(annotate(image, labels, footer))
    finally:
        writer.close()
    metadata = dict(source=str(capture), frames=len(rows), fps=fps, size=list(size),
                    backend=writer.backend, task=summary)
    Path(output).with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return dict(output=str(Path(output).resolve()), **metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--output", type=Path, help="Still directory (PNG montages)")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--video", type=Path, help="Write the three-view video here")
    parser.add_argument("--video-fps", type=float, help="Video frame rate (default: capture FPS)")
    parser.add_argument("--video-limit", type=int, help="Encode only the first N frames")
    args = parser.parse_args(argv)
    run = args.run
    if run is None:
        if args.run_id is None:
            raise SystemExit("Pass a run directory or --run-id")
        run = Storage.create(args.storage_root).run_dir(args.run_id)
    if args.video:
        metadata = build_video(run, args.video, fps=args.video_fps, limit=args.video_limit)
        print(f"VIDEO {metadata['output']} frames={metadata['frames']} "
              f"fps={metadata['fps']:.0f} backend={metadata['backend']}")
    if args.video is None or args.output is not None:
        output, written = build(run, args.output, args.count)
        print(f"PREVIEW {output} frames={len(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
