# Hardware

3D-printable parts and the full CAD assembly of the delta robot. This is not a step-by-step build
guide; see [photos.md](photos.md) and the videos in `media/` for how the parts go together.

## `stl/`: printed parts

| File | Part |
|---|---|
| `base_part1.stl`, `base_part2.stl` | Base, in two pieces |
| `upper_arm_6in_12mm.stl` | Upper arm driven by each servo ("Shoulder6in(12mm)" in the CAD); print 3 |
| `end_effector_electromagnet.stl` | End-effector platform with the electromagnet mount |
| `camera_holder.stl` | Camera mount |
| `camera_holder_base_mount.stl` | Camera mount that attaches to the base |

## `onshape_export/`

The Onshape assembly exported as **URDF + glTF meshes** (`assembly_1/urdf/assembly_1.urdf`, with a
ROS launch file). Delta robots are closed kinematic chains, so the URDF contains `*_loop_closure`
links from Onshape's export. Simulators that need a tree (e.g. plain ROS `robot_state_publisher`)
will only show the open part of each chain.

## Electronics (as built)

| Part | Notes |
|---|---|
| Teensy 4.1 | Runs `firmware/delta_teleop/delta_teleop.ino` |
| 3× Feetech STS3215 | Serial-bus servos on `Serial1` (TX pin 1 / RX pin 0), 1 Mbit/s, IDs 2, 3, 1 |
| 3× AS5600 magnetic encoder | Leader arm (teleoperation only). Two hardware I²C buses (`Wire`, `Wire2`) + one bit-banged bus on pins 20/21 |
| Electromagnet (~12 V) + H-bridge | IN1 = pin 30, IN2 = pin 32 |
| Push button | Magnet control during teleoperation, between pins 4 and 5 |
| 3× Innomaker U20CAM-720P | USB cameras on one hub, captured as MJPG 320×240 @ 30 fps |
