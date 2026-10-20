# NCA Lab 0 — Growing and repairing a pattern

This is a trainable **Neural Cellular Automaton** (NCA) built with a small shared local update network. A 12-channel state (RGBA + 8 hidden channels) is updated using depthwise Sobel neighborhood perception, random asynchronous updates and an alive-cell gate. The supervised target is a procedural flower.

**Status:** core and short training smoke-tested; successful long-run growth/regeneration is NOT yet validated. This is not a cloth solver, a physics engine or a general intelligent agent.

## Run on Windows PowerShell

Open this directory: book/chapters/chapter5/exp

1. Create environment: py -m venv .venv
2. Activate: .\.venv\Scripts\Activate.ps1
3. Install: pip install -r requirements.txt
4. Test: python -m pytest -q
5. Train: python train.py --steps 1200
6. Interact: python play.py

Use python train.py --help for tuning, --device cuda if PyTorch has CUDA support. On Linux/macOS activate via source .venv/bin/activate. Training writes model.pt, preview.png and train_summary.json locally.

### Controls
- Left mouse drag: erase cells
- Right click: implant seed
- Buttons: reset, damage half, pause/resume
- Dropdown: view RGB, alpha, or hidden channels

## Experimental design

Use the small model to learn how local memory and repeated updates can generate persistent spatial patterns. Training includes a pool of previously grown states and damaged states. Measure reconstruction after 32/64 steps, damage recovery after 16/32/64 steps, long-term survival, inference cost and random seed sensitivity. Compare with simpler fixed local rules before claiming emergent capability.

The flower is a target provided in the loss; its emergence is not unsupervised. The small-run preview is not proof that self-repair emerged.

## Literature and attribution

Inspired by Mordvintsev, Randazzo, Niklasson and Levin, Growing Neural Cellular Automata, Distill (2020): https://distill.pub/2020/growing-ca/

Upstream educational notebooks: https://github.com/google-research/self-organising-systems

Independent educational reimplementation, not a novel NCA algorithm. Public repository currently has no explicit license.

## Possible next experiments

1. Add localized intervention signals (moving obstacles or touches).
2. Compare uniform random updates to compute-budgeted selective updates.
3. Study information propagation, damage recovery and near-critical behavior.
4. Integrate active sensing before attempting non-LLM agents.
