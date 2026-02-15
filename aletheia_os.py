# aletheia_os.py
# Core foundation for Project Aletheia - A Standalone AR OS for Raspberry Pi
# SFHacks 2026 - Merged and Refactored Version

import pygame
import threading
import time
import os
import atexit
import numpy as np
import cv2
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Process, Queue as MPQueue, Event as MPEvent, Value

# Change E: prefer kmsdrm for direct GPU access when running from a TTY (not desktop).
# Only set if no display server is active — kmsdrm fails under X11/Wayland.
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    os.environ.setdefault("SDL_VIDEODRIVER", "kmsdrm")


def resolve_model_path(filename: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.join(base, filename),
        os.path.join(base, "models", filename),
        os.path.join(base, "meta-yolo", filename),
        os.path.join(base, "meta_yolo", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"[ModelPath] Using {p}")
            return p

    print("[ModelPath] ERROR: model file not found. Tried:")
    for p in candidates:
        print("  -", p)
    return candidates[0]

# Feature flags
# Run YOLO + GUI only:
#   ALETHEIA_ENABLE_HANDS=0 python aletheia_os.py
ENABLE_HANDS = os.getenv("ALETHEIA_ENABLE_HANDS", "1") == "1"

# RPi-specific camera controller (now also handles webcam fallback)
from camera_rpi import get_camera_manager

# GUI Components (repo-accurate: most widgets are draw-only; Spirit is a Sprite)
from aletheia_gui import SpiritCompanion, DetectionOverlay, HealthBar, MissionTracker, CarbonSavingsWidget

# YOLO worker — runs in a separate process to avoid GIL contention
from yolo_worker import yolo_worker_fn


# --- Global Constants (Change H: configurable via env vars) ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
FPS = int(os.getenv("ALETHEIA_FPS", "30"))

# BlazePalm / hand detector model (.pte)
BLAZEPALM_MODEL_PATH = resolve_model_path("blazepalm_xnnpack.pte")

# YOLO Object Detection model path
YOLO_MODEL_PATH = resolve_model_path("yolo26n_xnnpack.pte")

SHOW_CAMERA_BG = os.getenv("ALETHEIA_SHOW_CAMERA", "0") == "1"

# --- YOLO Multiprocessing Constants ---
FRAME_SHAPE = (720, 1280, 3)   # H, W, C — must match camera resolution
FRAME_DTYPE = np.uint8
FRAME_NBYTES = int(np.prod(FRAME_SHAPE)) * np.dtype(FRAME_DTYPE).itemsize
YOLO_INPUT_SIZE = int(os.getenv("ALETHEIA_YOLO_INPUT_SIZE", "640"))
YOLO_TARGET_HZ = float(os.getenv("ALETHEIA_YOLO_HZ", "10"))

# --- Shared State Dictionary (matches aletheia_gui.py expectations) ---
shared_state = {
    "is_pinching": False,
    "index_finger_tip": (0, 0),

    # GUI expects this name:
    "detected_objects": [],

    # SpiritCompanion reads these:
    "energy_waste_count": 0,
    "last_savings_event": "",
    "last_savings_event_time": 0.0,

    # Other HUD widgets read these:
    "health": 100,
    "missions_completed": 0,
    "missions_total": 5,
    "carbon_saved_g": 0.0,

    "app_quit": False,
}

state_lock = threading.Lock()


# --- HandTrackingThread (ExecuTorch BlazePalm detector, optional) ---
class HandTrackingThread(threading.Thread):
    """
    Optional ExecuTorch hand box detector (.pte).
    Provides:
      - shared_state["index_finger_tip"]: cursor point in screen coords
      - shared_state["is_pinching"]: currently always False
    """
    def __init__(self, model_path, camera, shared_state, state_lock,
                 input_size=256, confidence=0.6, target_hz=18, smoothing=0.35):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.camera = camera
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.input_size = int(input_size)
        self.confidence = float(confidence)
        self.target_hz = float(target_hz)
        self.smoothing = float(smoothing)
        self.detector = None
        self._cursor_f = None

    def run(self):
        print("[HandTrackingThread] Starting (ExecuTorch BlazePalm)...")

        if not os.path.exists(self.model_path):
            print(f"[HandTrackingThread] WARNING: BlazePalm .pte not found at {self.model_path}")
            print("[HandTrackingThread] Hand tracking disabled — YOLO + GUI will continue.")
            return

        try:
            from blazepalm_engine import BlazePalmDetector  # lazy import
            self.detector = BlazePalmDetector(
                self.model_path,
                input_size=self.input_size,
                confidence_threshold=self.confidence,
            )
        except Exception as e:
            print(f"[HandTrackingThread] WARNING: could not load BlazePalm model: {e}")
            print("[HandTrackingThread] Hand tracking disabled — YOLO + GUI will continue.")
            return

        print("[HandTrackingThread] Waiting for camera frames...")
        period = 1.0 / max(self.target_hz, 0.1)
        next_t = time.time()

        while True:
            with self.state_lock:
                if self.shared_state["app_quit"]:
                    break

            now = time.time()
            if now < next_t:
                time.sleep(min(0.002, next_t - now))
                continue

            next_t += period
            if now - next_t > 0.5:
                next_t = now + period

            frame_rgb = self.camera.get_frame()
            if frame_rgb is None:
                continue

            frame_rgb = cv2.flip(frame_rgb, 1)

            try:
                dets = self.detector.detect(frame_rgb)
            except Exception as e:
                print(f"[HandTrackingThread] ERROR during detect: {e}")
                dets = []

            cursor = None
            pinch = False  # placeholder until landmarks exist

            if dets:
                best = dets[0]
                h, w = frame_rgb.shape[:2]
                cx, cy = best["center"]

                x1, y1, x2, y2 = best["box"]
                cy = int(y1 + (y2 - y1) * 0.25)

                sx = float(cx) * SCREEN_WIDTH / max(w, 1)
                sy = float(cy) * SCREEN_HEIGHT / max(h, 1)
                cursor = (sx, sy)

            if cursor is None:
                cursor = self._cursor_f if self._cursor_f is not None else (0.0, 0.0)

            if self._cursor_f is None:
                self._cursor_f = cursor
            else:
                a = self.smoothing
                self._cursor_f = (
                    self._cursor_f[0] * a + cursor[0] * (1.0 - a),
                    self._cursor_f[1] * a + cursor[1] * (1.0 - a),
                )

            cursor_int = (int(self._cursor_f[0]), int(self._cursor_f[1]))

            with self.state_lock:
                self.shared_state["index_finger_tip"] = cursor_int
                self.shared_state["is_pinching"] = pinch

        print("[HandTrackingThread] Stopped.")


def main():
    multiprocessing.set_start_method("fork", force=True)
    pygame.init()
    print(f"[Main] SDL video driver: {pygame.display.get_driver()}")
    print(f"[Main] FPS target: {FPS} (set ALETHEIA_FPS to change)")

    # Fullscreen (no borders), use the display's native resolution
    flags = pygame.FULLSCREEN | pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF
    screen = pygame.display.set_mode((0, 0), flags)

    # Update constants to actual fullscreen resolution
    SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()
    print(f"[Main] Display resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

    pygame.mouse.set_visible(False)
    pygame.display.set_caption("Aletheia OS")

    clock = pygame.time.Clock()

    # Init GUI Components (repo-accurate constructors)
    spirit = SpiritCompanion(shared_state, state_lock)
    spirit_group = pygame.sprite.Group(spirit)

    overlay = DetectionOverlay(shared_state, state_lock)
    health_bar = HealthBar(shared_state, state_lock)
    mission_tracker = MissionTracker(shared_state, state_lock)
    carbon_widget = CarbonSavingsWidget(shared_state, state_lock)

    print("[Main] Initializing Camera...")
    camera = get_camera_manager()
    camera.start()
    print("[Main] Camera started.")

    # --- YOLO detection in a separate process (GIL-free) ---
    # Clean up stale shared memory from a previous crash, if any
    try:
        _stale = SharedMemory(name="aletheia_frame", create=False)
        _stale.close()
        _stale.unlink()
        print("[Main] Cleaned up stale shared memory from previous run.")
    except FileNotFoundError:
        pass
    shm = SharedMemory(name="aletheia_frame", create=True, size=FRAME_NBYTES)

    def _cleanup_shm():
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass
    atexit.register(_cleanup_shm)

    frame_seq = Value('q', 0)
    yolo_result_queue = MPQueue(maxsize=2)
    yolo_stop_event = MPEvent()

    yolo_proc = Process(
        target=yolo_worker_fn,
        args=(
            YOLO_MODEL_PATH,
            shm.name,
            FRAME_SHAPE,
            str(FRAME_DTYPE),
            frame_seq,
            yolo_result_queue,
            yolo_stop_event,
        ),
        kwargs={
            "input_size": YOLO_INPUT_SIZE,
            "target_hz": YOLO_TARGET_HZ,
        },
        daemon=True,
    )
    yolo_proc.start()
    print(f"[Main] YOLO worker process started (PID={yolo_proc.pid}, "
          f"input_size={YOLO_INPUT_SIZE}, target_hz={YOLO_TARGET_HZ})")

    # Numpy view into shared memory for zero-copy frame writes
    _shm_array = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm.buf[:FRAME_NBYTES])

    # Start Hand thread (optional)
    if ENABLE_HANDS:
        hand_thread = HandTrackingThread(
            model_path=BLAZEPALM_MODEL_PATH,
            camera=camera,
            shared_state=shared_state,
            state_lock=state_lock,
            target_hz=18
        )
        hand_thread.start()
    else:
        print("[Main] Hand tracking disabled (ALETHEIA_ENABLE_HANDS=0). Running YOLO+GUI only.")

    print("[Main] Entering main loop...")

    running = True
    cam_surface = None
    cam_update_every = 3  # update camera texture every N GUI frames
    cam_counter = 0

    # Frame feeder: copy camera frame to shared memory for YOLO worker
    _feed_every = max(1, int(FPS / YOLO_TARGET_HZ))
    _feed_counter = 0

    # --- Instrumentation (Change A) ---
    _perf_enabled = os.getenv("ALETHEIA_PERF_LOG", "1") == "1"
    _perf_file = None
    _frame_num = 0
    if _perf_enabled:
        try:
            _perf_file = open("/tmp/aletheia_perf.csv", "w")
            _perf_file.write("frame,t_event,t_snapshot,t_update,t_fill,t_camera,t_widgets,t_flip,t_total,fps\n")
            print("[Perf] Logging to /tmp/aletheia_perf.csv (set ALETHEIA_PERF_LOG=0 to disable)")
        except OSError:
            _perf_enabled = False

    # FPS console print timer (Change J)
    _next_fps_print = time.time() + 2.0

    while running:
        # Change C: dt from clock.tick (milliseconds -> seconds)
        dt_ms = clock.tick(FPS)
        dt = min(dt_ms / 1000.0, 0.1)  # clamp to prevent huge jumps

        _t0 = time.perf_counter()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        _t_event = time.perf_counter()

        # If hands disabled, use mouse as cursor/pinch for testing
        if not ENABLE_HANDS:
            mx, my = pygame.mouse.get_pos()
            with state_lock:
                shared_state["index_finger_tip"] = (mx, my)
                shared_state["is_pinching"] = pygame.mouse.get_pressed()[0]

        # Feed camera frame to YOLO worker via shared memory
        _feed_counter += 1
        if _feed_counter >= _feed_every:
            _feed_counter = 0
            feed_frame = camera.get_frame()
            if feed_frame is not None and feed_frame.shape == FRAME_SHAPE:
                np.copyto(_shm_array, feed_frame)
                frame_seq.value += 1

        # Drain YOLO detection results from worker process (non-blocking)
        try:
            while True:
                result = yolo_result_queue.get_nowait()
                if isinstance(result, dict) and "error" in result:
                    print(f"[Main] YOLO worker error: {result['error']}")
                    running = False
                    break
                with state_lock:
                    shared_state["detected_objects"] = result
        except Exception:
            pass  # queue.Empty is expected

        # --- Single state snapshot for this frame ---
        with state_lock:
            state_snapshot = dict(shared_state)

        _t_snapshot = time.perf_counter()

        # Update Spirit (pass snapshot + dt for frame-rate-independent animation)
        spirit_group.update(state_snapshot, dt)

        _t_update = time.perf_counter()

        # Draw
        screen.fill((0, 0, 0))

        _t_fill = time.perf_counter()

        # Camera background (throttled conversion with optimized pipeline)
        cam_counter += 1
        if cam_counter % cam_update_every == 0:
            frame = camera.get_frame()
            if frame is not None:
                resized = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_NEAREST)
                cam_surface = pygame.surfarray.make_surface(np.rot90(resized))

        if SHOW_CAMERA_BG and cam_surface is not None:
            screen.blit(cam_surface, (0, 0))

        _t_camera = time.perf_counter()

        # Pass state_snapshot to all widgets
        overlay.draw(screen, state_snapshot)
        spirit_group.draw(screen)
        health_bar.draw(screen, state_snapshot)
        mission_tracker.draw(screen, state_snapshot)
        carbon_widget.draw(screen, state_snapshot)

        _t_widgets = time.perf_counter()

        pygame.display.flip()

        _t_flip = time.perf_counter()

        # --- Instrumentation logging (Change A) ---
        _frame_num += 1
        if _perf_enabled and _perf_file and _frame_num % 60 == 0:
            _fps = clock.get_fps()
            _perf_file.write(
                f"{_frame_num},"
                f"{_t_event - _t0:.6f},{_t_snapshot - _t_event:.6f},"
                f"{_t_update - _t_snapshot:.6f},{_t_fill - _t_update:.6f},"
                f"{_t_camera - _t_fill:.6f},{_t_widgets - _t_camera:.6f},"
                f"{_t_flip - _t_widgets:.6f},{_t_flip - _t0:.6f},{_fps:.1f}\n"
            )
            _perf_file.flush()

        # --- FPS console print (Change J) ---
        _now = time.time()
        if _now >= _next_fps_print:
            _next_fps_print = _now + 2.0
            print(f"[Perf] FPS: {clock.get_fps():.1f}  frame: {(_t_flip - _t0) * 1000:.1f}ms"
                  f"  update: {(_t_update - _t_snapshot) * 1000:.1f}ms"
                  f"  flip: {(_t_flip - _t_widgets) * 1000:.1f}ms")

    print("[Main] Shutting down...")
    with state_lock:
        shared_state["app_quit"] = True

    # Stop YOLO worker process
    yolo_stop_event.set()
    yolo_proc.join(timeout=3)
    if yolo_proc.is_alive():
        print("[Main] YOLO worker did not exit cleanly, terminating...")
        yolo_proc.terminate()
        yolo_proc.join(timeout=1)

    # Cleanup shared memory
    try:
        shm.close()
        shm.unlink()
    except Exception as e:
        print(f"[Main] SharedMemory cleanup warning: {e}")

    if _perf_file:
        _perf_file.close()

    camera.stop()
    pygame.quit()
    print("[Main] Goodbye.")


if __name__ == "__main__":
    main()
