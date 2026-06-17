# Bee Colony Simulation

Agent-based simulation of a honeybee colony built with Python and [Mesa](https://mesa.readthedocs.io/).
Developed for the *Agent Systems* course at AGH University of Science and Technology (2025/2026).

## Features

- **4 agent types:** Queen, Nurse, Forager, Scout
- **HoPoMo role switching** (Schmickl & Crailsheim 2007): nurses transition to foragers by age or colony nectar deficit
- **Waggle dance recruitment:** returning foragers recruit resting foragers proportional to patch profitability
- **Pheromone Cellular Automaton:** vectorised NumPy diffusion + decay layer with stigmergic trail formation
- **RL forager (optional):** PPO policy trained with stable-baselines3 as an alternative to the hand-crafted FSM
- **Live visualisation:** matplotlib display with pause/resume, speed slider, and real-time trend charts
- **Headless batch mode:** `--batch N` runs N independent seeds and prints a comparison table
- **Parameter sweep:** `scripts/sweep.py` explores nurse age, patch count, and trail strength combinations

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Running

```bash
# Interactive (opens mode selection window)
python -m src.run

# Headless, 300 steps
python -m src.run --headless --steps 300

# Batch run, 10 seeds
python -m src.run --batch 10 --steps 300 --seed 0

# Record to video
python -m src.run --record sim.mp4 --steps 500

# RL forager mode
python -m src.run --rl-foragers

# Disable pheromones
python -m src.run --no-pheromones
```

## Simulation modes

| Mode | Description |
|---|---|
| FSM + pheromones | Default: hand-crafted FSM foragers with pheromone CA |
| FSM no pheromones | FSM foragers, pheromone layer disabled |
| RL + pheromones | PPO-controlled foragers replace the FSM |

## Project structure

```
src/
  config.py              all tunable parameters
  model.py               BeeColonyModel (Mesa Model)
  run.py                 CLI entry point
  visualization.py       matplotlib live display
  agents/
    base.py              BeeAgent (Mesa 1.x/2.x compatibility shim)
    queen.py             QueenAgent
    nurse.py             NurseAgent (HoPoMo role switching)
    forager.py           ForagerAgent (5-state FSM)
    scout.py             ScoutAgent
    rl_forager.py        RLForagerAgent (PPO policy)
  environment/
    hive.py              Hive data class
    flower_patch.py      FlowerPatch data class
scripts/
  sweep.py               parameter sweep (exports CSV)
  plot_sweep.py          generates figures from sweep CSV
rl_forager_mvp/
  env.py                 single-agent Gymnasium environment (stage 1 training)
  colony_env.py          full-colony Gymnasium environment (stage 2 training)
  train_rl.py            stage 1 PPO training
  train_colony_rl.py     stage 2 PPO training
  eval_rl.py             policy evaluation script
  models/
    ppo_forager_colony.zip  trained PPO policy
```

## Key parameters (`src/config.py`)

| Parameter | Default | Effect |
|---|---|---|
| `NUM_FLOWER_PATCHES` | 12 | Number of flower patches on the grid |
| `NURSE_TO_FORAGER_AGE` | 60 | Steps before natural nurse-to-forager switch |
| `TRAIL_DEPOSIT_STRENGTH` | 0.6 | Pheromone deposited per step on return trip |
| `PHEROMONE_DECAY` | 0.997 | Per-step pheromone retention (half-life ~231 steps) |
| `WAGGLE_RECRUIT_MAX` | 5 | Max foragers recruited per waggle dance |

## Authors

Jan Kusa, Oliwier Maj, Michał Krzempek

AGH University of Science and Technology, Faculty of EAIiIB

| Name | Student ID | Email |
|---|---|---|
| Jan Kusa | 414587 | jankusa@student.agh.edu.pl |
| Oliwier Maj | 415269 | majoliwier@student.agh.edu.pl |
| Michał Krzempek | 417591 | krzempekm@student.agh.edu.pl |
