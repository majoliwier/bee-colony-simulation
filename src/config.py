# ── Grid ────────────────────────────────────────────────────────────────────
GRID_WIDTH = 50
GRID_HEIGHT = 50

# ── Hive ────────────────────────────────────────────────────────────────────
HIVE_POS = (25, 25)
INITIAL_NECTAR = 600.0   # ~60 % full — colony starts healthy, nurses won't panic-switch
MAX_NECTAR_STORES = 1000.0

# ── Queen ───────────────────────────────────────────────────────────────────
EGG_INTERVAL = 10           # steps between egg-laying events
EGGS_PER_INTERVAL_MAX = 3   # max eggs per event; actual count scales with nectar ratio
EGG_INCUBATION_STEPS = 21   # steps until egg hatches into a nurse

# ── Nurse ───────────────────────────────────────────────────────────────────
INITIAL_NURSES = 15
NURSE_FEEDING_RATE = 0.3         # nectar consumed per nurse per step (brood feeding)
NURSE_TO_FORAGER_AGE = 60        # age (steps) at which nurse naturally becomes forager
# HoPoMo: each nurse's personal deficit threshold is drawn from this range at birth.
# If colony deficit exceeds the threshold, the nurse switches early.
NURSE_FORAGER_THRESHOLD_RANGE = (0.3, 0.7)

# ── Forager ─────────────────────────────────────────────────────────────────
INITIAL_FORAGERS = 8
FORAGER_REST_DURATION = 3    # steps spent resting in hive after each return
FORAGER_LOAD_CAPACITY = 10.0  # max nectar a forager can carry
COLLECTING_STEPS = 3          # steps the forager stays on a patch to fill its load

# ── Flower Patches ──────────────────────────────────────────────────────────
NUM_FLOWER_PATCHES = 12          # more patches → foragers find them faster during random walk
PATCH_MAX_NECTAR = 80.0
PATCH_REGEN_RATE = 0.3          # nectar regenerated per step
PATCH_QUALITY_RANGE = (0.5, 1.5)  # quality multiplier range (applied to collected nectar)
MIN_PATCH_DISTANCE = 5           # allow patches closer to hive so early demos show activity

# ── Simulation / Visualisation ───────────────────────────────────────────────
DEFAULT_STEPS = 300
VIZ_UPDATE_INTERVAL = 3  # redraw display every N steps
