# ── Grid ────────────────────────────────────────────────────────────────────
GRID_WIDTH = 50
GRID_HEIGHT = 50

# ── Hive ────────────────────────────────────────────────────────────────────
HIVE_POS = (25, 25)
INITIAL_NECTAR = 600.0   # ~60% full at start
MAX_NECTAR_STORES = 1000.0

# ── Queen ───────────────────────────────────────────────────────────────────
EGG_INTERVAL = 10           # steps between egg-laying events
EGGS_PER_INTERVAL_MAX = 3   # max eggs per event; actual count scales with nectar ratio
EGG_INCUBATION_STEPS = 21   # steps until egg hatches into a nurse

# ── Nurse ───────────────────────────────────────────────────────────────────
INITIAL_NURSES = 15
NURSE_FEEDING_RATE = 0.3         # nectar consumed per nurse per step (brood feeding)
NURSE_TO_FORAGER_AGE = 60        # age (steps) at which nurse naturally becomes forager
NURSE_FORAGER_THRESHOLD_RANGE = (0.3, 0.7)  # personal deficit threshold drawn at birth (HoPoMo)

# ── Forager ─────────────────────────────────────────────────────────────────
INITIAL_FORAGERS = 8
FORAGER_REST_DURATION = 6    # steps spent resting in hive after each return
FORAGER_LOAD_CAPACITY = 10.0  # max nectar a forager can carry
COLLECTING_STEPS = 3          # steps the forager stays on a patch to fill its load
USE_RL_FORAGERS = False       # False = FSM foragers, True = PPO-controlled foragers
RL_FORAGER_MODEL_PATH = "rl_forager_mvp/models/ppo_forager_colony.zip"
RL_FORAGER_REST_DURATION = 0  # PPO was trained without mandatory post-trip rest

# ── Flower Patches ──────────────────────────────────────────────────────────
NUM_FLOWER_PATCHES = 12
PATCH_MAX_NECTAR = 80.0
PATCH_REGEN_RATE = 0.3          # nectar regenerated per step
PATCH_QUALITY_RANGE = (0.5, 1.5)  # quality multiplier range (applied to collected nectar)
MIN_PATCH_DISTANCE = 5           # minimum Manhattan distance from hive
PATCH_LIFETIME_NECTAR = 200.0    # total nectar a patch can ever yield before dying permanently

# ── Forager energy / death ──────────────────────────────────────────────────
FORAGER_MAX_ENERGY = 200         # steps of flight before exhaustion
FORAGER_ENERGY_COST_PER_STEP = 1 # energy lost per step while not resting
LOCAL_SEARCH_STEPS = 35          # random-walk steps after missing dance target; 0 -> give up
SMELL_RADIUS = 2                 # Chebyshev distance at which a bee can detect a flower patch

# ── Nurse death ──────────────────────────────────────────────────────────────
MAX_NURSE_AGE = 100              # fallback death age if role switch never triggers

# ── Waggle dance ─────────────────────────────────────────────────────────────
WAGGLE_RECRUIT_MAX = 5           # max foragers recruited per dance
WAGGLE_PROFITABILITY_SCALE = 1.0 # profitability at which max recruits are triggered (~close full patch)
WAGGLE_ANGLE_NOISE = 0.25        # std dev of angular error in dance communication (radians, ~14 deg)

# ── Scout ────────────────────────────────────────────────────────────────────
INITIAL_SCOUTS = 5
SCOUT_MAX_ENERGY = 300           # higher than foragers; scouts never stop to collect

# ── Pheromone (Cellular Automaton) ───────────────────────────────────────────
# half-life ~231 steps: phi_{t+1} = (phi_t*(1-D) + D*avg_neighbours) * decay
PHEROMONE_DECAY        = 0.997   # per-step retention fraction
PHEROMONE_DIFFUSION    = 0.015   # fraction spread to Moore neighbours
TRAIL_DEPOSIT_STRENGTH = 0.6     # deposit per step on outbound flight
PHEROMONE_BIAS         = 0.85    # weight of pheromone vs random in scouting movement

# ── Simulation / Visualisation ───────────────────────────────────────────────
DEFAULT_STEPS = 300
VIZ_UPDATE_INTERVAL = 3  # redraw display every N steps
