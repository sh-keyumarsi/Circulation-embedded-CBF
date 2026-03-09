# Circulation-embedded Control Barrier Function for safe navigation: A solution to avoid undesired equilibria and dysfunctional circulation
This repository provides a simulation reproducing the experiments from https://doi.org/10.1016/j.robot.2025.105132, including the simulator, source code, and data.  Contains two scenarios, nonconvex obstacle avoidance and multi-agent deadlock resolution using IBCECBFQP, each runnable via lidar_gp_cbf/sim2D_main.


Code and simulation assets for the paper:
> **Circulation-embedded Control Barrier Function for safe navigation: A solution to avoid undesired equilibria and dysfunctional circulation**  
> Shaghayegh Keyumarsi et al.


## Experiment

You can view the experiment here:  
[NeboLab Experiment](https://www.youtube.com/watch?v=69W5bmbBVcI)

# Circulation-Embedded CBF — Code & Simulations

#Simulation

This repository includes a simulator that visualizes the robotic experiment as the code runs.
When executed, the simulation window will open and display the robot’s behavior step by step.


How to run:

-Nonconvex obstacle simulation

python nonconvex-sim/lidar_gp_cbf/sim2D_main.py

-Multi-agent deadlock resolution

python multiagent-deadlock-sim/lidar_gp_cbf/sim2D_main.py


Outputs & animations
Generated media are saved under each module’s animation_result/ (tracked in Git).

Re-running scripts may overwrite existing files.

Contact
Shaghayegh Keyumarsi — GitHub: sh-keyumarsi
