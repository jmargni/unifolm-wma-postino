# PEP demo script — the twin's policy checks on the packing loop

A presenter's walkthrough for showing the DGS G1-D Cyber Twin's **Policy
Enforcement Point (PEP)** checking the model's commands while the robot packs
the camera. Everything runs in the MuJoCo simulator; no physical robot.

Four scenarios, each about a minute, all using the controls the twin console
already has. No extra code. They show the same point from four angles: the
model drives the robot only through the PEP, and an operator can stop or
constrain it at any time, with every decision logged.

## Setup (once, before the audience)

Setup B from the README, with the replay server looping so the task repeats on
its own. Three terminals in `unitree_deploy/scripts`:

```bash
python sim_g1_robot.py --robot g1d --twin_url http://127.0.0.1:3000 --auto_reset --headless --web_port 8080
python replay_policy_server.py --cover --loop
cd ~/projects/DGS-CyberTwin-G1D && python server.py        # the twin (PEP + consoles)
UNITREE_IMAGE_SERVER=127.0.0.1 python robot_client.py --control_freq 15 --pep_url http://127.0.0.1:3000
```

> On a shared server where port 5555 is taken, add `--image_port 5560` to the
> simulator and `UNITREE_IMAGE_PORT=5560` to the client (see the README).

`--headless`: on the demo server there is no screen for the MuJoCo window (without
it the simulator would stop at start-up and the postino would wait at
`Waiting to subscribe dds...`); the robot is shown in the web view instead.

Two browser windows, side by side:

- **The robot** — the web view, `http://<server>:8080`.
- **The twin console** — `http://localhost:3000`. Use its left nav to switch
  between *Digital twin* (robot controls), *Defense* (policy, identities),
  *SIEM & audit* (the decision log) and *Test harness*.

Keep the **client's terminal** visible too: it prints one line per chunk,
`PEP chunk: ALLOW G1D-200`, so the audience sees the check happen in real time.

**Start the loop:** in the *Digital twin* page, click **Arma robot**. The
client stops printing `PEP start pose: DENY` and begins the task; the robot
picks up the camera. Until it is armed, nothing moves — that is the first point
to make: the model's commands wait for a human to authorise the robot.

## Scenario 1 — Emergency stop mid-motion

**Goal:** an operator halts the robot instantly, whatever the model is asking.

1. While the right arm is carrying the camera, press the red **E-STOP** button
   (top right of the twin console).
2. **Audience sees:** the robot freezes mid-motion in the simulator. The
   client's terminal prints `PEP: twin no longer armed (E-stop, disarm or
   unreachable): chunk aborted`. The console's robot state turns to `estop`.
3. **Say:** "The model is still sending the next move, but the PEP checks the
   robot is armed before every single step. One operator action overrides the
   AI completely — it doesn't negotiate."
4. **Resume:** on the *Defense* page, click **Ripristina twin → disarmed**,
   then **Arma robot** again on the *Digital twin* page. The loop continues on
   its next pass.

## Scenario 2 — Revoking the model's identity

**Goal:** the commands carry an identity; revoke it and the PEP stops obeying.

1. Go to the *Defense* page. Under the identities, pick **operator** in the
   list and click **Revoca**.
2. **Audience sees:** the robot holds its pose. The client's terminal switches
   from `PEP chunk: ALLOW G1D-200` to `PEP chunk: DENY G1D-101 ...`. On the
   *SIEM & audit* page, DENY rows appear with rule **G1D-101** (unauthorized).
3. **Say:** "Every command the model sends is signed as `operator`. Revoke that
   identity — as you would a leaked key — and the PEP refuses the commands at
   once. The robot doesn't move on stale authority."
4. **Resume:** click **Ripristina identità**. The next chunk is allowed again
   and the robot carries on.

## Scenario 3 — Monitor vs enforce

**Goal:** show the difference between blocking a violation and only logging it.

1. Still on the *Defense* page, set the **Modalità PEP** to **Monitor · solo
   valutazione**.
2. Repeat scenario 2: revoke **operator**.
3. **Audience sees:** this time the robot **keeps moving**. On the *SIEM &
   audit* page the same violation is logged, but as **MONITOR**, not DENY.
4. **Say:** "Monitor mode is how you roll a new policy out safely: you see
   exactly what it *would* have blocked, on real traffic, before you switch it
   to enforce. Here the revoked identity is recorded but not acted on."
5. **Resume:** click **Ripristina identità**, then set the mode back to
   **Enforce · DENY**.

## Scenario 4 — The operator still drives the base

**Goal:** the PEP gates the operator's own commands too, not only the model's.

1. On the *Digital twin* page, use the **BASE AGV** pad (the arrows) or the
   **Estensione colonna** slider + **Applica altezza**.
2. **Audience sees:** the simulated robot's base moves (or the column rises) to
   match the console, once the PEP allows the command. The *SIEM & audit* page
   logs each one as ALLOW **G1D-200**.
3. **Say:** "The base and column move through the same PEP as the model's arm
   commands. Every actor — the AI and the human — goes through one checkpoint,
   and one audit log. Nothing reaches the robot around it."

> Base and column commands move the whole robot, so run this during a loop
> pause (between packs) rather than mid-task, or it drives the robot away from
> the table.

## If you want to show the policy catalogue

The *Red Team* page lists eight attack scenarios from the threat model, and the
*Test harness* page runs them all at once (**Esegui suite completa** →
9/9 passed). These are sent to the twin's own virtual robot: each is denied
with its rule (G1D-101…108), proving the PEP rejects the request *before* it
could reach any robot. Good as a closing slide: "these are the classes of
attack the checkpoint stops."

## Reset between runs

- Robot stuck or in a bad pose: **Ripristina twin → disarmed** (Defense), then
  **Arma robot** (Digital twin).
- Objects out of place: the simulator puts them back on its own 3 s after the
  arms return to the start pose (`--auto_reset`), or press **reset scene** in
  the web view.
