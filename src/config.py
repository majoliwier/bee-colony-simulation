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
FORAGER_REST_DURATION = 6    # steps spent resting in hive after each return
FORAGER_LOAD_CAPACITY = 10.0  # max nectar a forager can carry
COLLECTING_STEPS = 3          # steps the forager stays on a patch to fill its load
USE_RL_FORAGERS = False       # False = FSM foragers, True = PPO-controlled foragers
RL_FORAGER_MODEL_PATH = "rl_forager_mvp/models/ppo_forager_colony.zip"
RL_FORAGER_REST_DURATION = 0  # PPO was trained without mandatory post-trip rest

# ── Flower Patches ──────────────────────────────────────────────────────────
NUM_FLOWER_PATCHES = 12          # more patches → foragers find them faster during random walk
PATCH_MAX_NECTAR = 80.0
PATCH_REGEN_RATE = 0.3          # nectar regenerated per step
PATCH_QUALITY_RANGE = (0.5, 1.5)  # quality multiplier range (applied to collected nectar)
MIN_PATCH_DISTANCE = 5           # allow patches closer to hive so early demos show activity
PATCH_LIFETIME_NECTAR = 200.0    # total nectar a patch can ever yield before dying permanently

# ── Forager energy / death ──────────────────────────────────────────────────
FORAGER_MAX_ENERGY = 200         # steps of flight before exhaustion
FORAGER_ENERGY_COST_PER_STEP = 1 # energy lost per step while not resting
LOCAL_SEARCH_STEPS = 35          # random-walk steps after missing dance target; 0 -> give up
SMELL_RADIUS = 2                 # Chebyshev distance at which a bee can detect a flower patch

# ── Nurse death ──────────────────────────────────────────────────────────────
MAX_NURSE_AGE = 100              # fallback death age (should normally switch first)

# ── Waggle dance ─────────────────────────────────────────────────────────────
WAGGLE_RECRUIT_MAX = 5           # max foragers recruited per dance
WAGGLE_PROFITABILITY_SCALE = 1.0 # profitability at which max recruits are triggered (~close full patch)
WAGGLE_ANGLE_NOISE = 0.25        # std dev of angular error in dance communication (radians, ~14 deg)

# ── Scout ────────────────────────────────────────────────────────────────────
INITIAL_SCOUTS = 5
SCOUT_MAX_ENERGY = 300           # scouts fly more (always random walk)

# Pheromone (Cellular Automaton)
# Effective local retention per step = DECAY * (1 - DIFFUSION).
# Half-life is about 38 steps, long enough for roughly one forager round-trip.
PHEROMONE_DECAY        = 0.997   # global decay fraction per step
PHEROMONE_DIFFUSION    = 0.015   # fraction spread to Moore neighbours
TRAIL_DEPOSIT_STRENGTH = 0.6     # deposit per step on outbound flight
PHEROMONE_BIAS         = 0.85    # weight of pheromone vs random in scouting movement

# ── Simulation / Visualisation ───────────────────────────────────────────────
DEFAULT_STEPS = 300
VIZ_UPDATE_INTERVAL = 3  # redraw display every N steps
