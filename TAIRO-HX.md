## **TAIRO-HX: Hierarchical Failure Detection and Recovery for SAC+HER** 

## **1. Overview** 

The current TAIRO system uses a SAC+HER policy for robot Pick-and-Place tasks. Under normal conditions, the policy performs well. However, cyber-physical attacks can change the robot’s actions, sensor readings, gripper state, object position, or goal. 

Recovery v4 adds an online Random Forest classifier using about 40 behavior-based features. It predicts whether the robot is operating normally or experiencing failures such as divergent motion, failed grasp, object-pose spoofing, or object drop. The classifier confidence is then used to blend recovery actions with the SAC+HER policy. Current results show that the system preserves clean performance and works reasonably well for action clipping and delay, but recovery is still weak for goal spoofing, grip-state falsification, action reversal, and sensor dropout. 

We propose an improved version called **TAIRO-HX** , which combines SAC+HER with a hierarchical classifier and a selective recovery controller. 

## **2. Main idea** 

The current classifier mixes together three different questions: 

- What attack happened? 

- What failure is the robot showing? 

- What recovery should be applied? 

These should be separated. 

For example, object-pose spoofing is an attack, while “never reached the object” is a failure caused by that attack. The recovery may then involve reconstructing the object pose and asking SAC+HER to try again. 

The new pipeline is: 

## **SAC+HER→Task Stage→Anomaly Detection→Failure Type→Attack Family→Recovery Decision** 

## **3. Hierarchical classifier** 

## **Level 1: Task stage** 

First, determine what the robot is currently doing: 

- Approaching the object 

- Aligning the gripper 

- Grasping 

- Transporting 

- Placing 

- Verifying completion 

The same behavior can mean different things at different stages. An open gripper is normal during approach but may indicate failure during transport. 

## **Level 2: Normal or abnormal** 

The system then decides whether the rollout is: 

- Normal 

- Suspicious 

- Abnormal 

- Unknown 

The classifier should look at several recent timesteps rather than one timestep at a time. Its probabilities can be smoothed to avoid rapidly switching between normal and recovery modes. 

## **Level 3: Behavioral failure** 

When abnormal behavior is detected, the classifier predicts what is going wrong: 

- Moving away from the object or goal 

- No progress 

- Reaching the wrong location 

- Failed grasp 

- False grasp confirmation 

- Dropped object 

- Moving toward the wrong goal 

- Missing or conflicting sensor information 

This level should support multiple labels because one attack may produce several symptoms. 

## **Level 4: Likely attack family** 

The system then estimates the likely source: 

- Action or actuation attack 

- Perception or state attack 

- Goal manipulation 

- Sensor-information loss 

- Unknown attack 

Exact attack identification is helpful, but recovery should not depend completely on it. The robot may know that the object position is unreliable without knowing whether the cause is spoofing, sensor bias, or tracking failure. 

## **Level 5: Recoverability** 

Finally, the system decides what to do: 

- Continue SAC+HER 

- Compensate for the problem 

- Reconstruct the state 

- Retry the task stage 

- Restore the trusted goal 

- Continue in a reduced-speed mode 

- Stop safely 

## **4. Classifier algorithms** 

Random Forest should remain as the main baseline because it is fast, interpretable, and already works with the current features. 

However, the project should also test: 

**Model Purpose** Logistic regression Simple baseline Random Forest Current interpretable baseline XGBoost Main proposed tabular classifier LightGBM Fast boosting baseline CatBoost Robust boosting baseline MLP Basic neural baseline GRU or TCN Temporal baseline The most useful comparison is: 

# **Flat RF vs. Flat XGBoost vs. Hierarchical RF vs. Hierarchical XGBoost.** 

This will show whether improvement comes from XGBoost, the hierarchy, or both. 



<!-- Start of picture text -->
aySAC __= TSAC+HER (0; 9):<br><!-- End of picture text -->



<!-- Start of picture text -->
are = a’,<br><!-- End of picture text -->

a,Tec = TSAC+HER (Ot,~ 9): 

- Loss of both vision and contact 

- Unknown attacks with high uncertainty 

- Repeated failed recovery attempts 

In these cases, stopping safely is better than continuing with an unreliable correction. 

## **6. Improved recovery families** 

The five current specialist rules can be reorganized into broader recovery families: 

1. Execution correction Handles clipping, delay, and bias. 

2. Trusted-state reconstruction Handles object-pose errors, false grip state, and conflicting sensors. 

3. Task retry Handles failed approach, failed grasp, dropped object, and failed placement. 

4. Goal restoration and re-planning Handles immediate and mid-episode goal spoofing. 

5. Safe degradation or stop Handles partial sensor loss, high uncertainty, and unrecoverable attacks. 

The system should avoid blending all recovery specialists at the same time. Some specialists may suggest conflicting actions. Instead, it should select one recovery family or blend only compatible actions. 

## **7. Key features** 

The classifier should use causal features from recent timesteps, including: 

- Distance to the object 

- Distance to the goal 

- Progress over 5, 10, and 20 steps 

- End-effector speed and direction 

- Action variance 

- Commanded versus executed action difference 

- Object–gripper distance 

- Whether the object follows the gripper 

- Gripper width and contact 

- Goal changes 

- Missing-sensor indicators 

- Number of previous recovery attempts 

The train and test sets should be divided by complete episodes or seeds. Randomly dividing individual timesteps could leak nearly identical neighboring observations into both sets. 

## **8. Evaluation** 

Classifier performance should include: 

- Macro-F1 

- Per-class precision and recall 

- Confusion matrix 

- False alarms on clean episodes 

- Detection delay 

- Calibration 

- Inference time 

Recovery evaluation should include: 

- Task-success rate 

- Improvement over no recovery 

- Clean-performance loss 

- Safe-stop rate 

- Unsafe completion rate 

- Number of interventions 

- Recovery time 

- Completion-time overhead 

The most important ablation is: 

1. No recovery 

2. Recovery with predicted failure labels 

3. Recovery with true failure labels 

4. Recovery with true failure labels and simulator state 

5. Recovery with predicted labels and deployable state estimates 

This will show whether the main bottleneck is the classifier, the state estimator, or the recovery controller. 

## **9. Recommended next steps** 

The team should first implement four changes: 

1. Compare hierarchical Random Forest and hierarchical XGBoost. 

2. Add task-stage and recoverability prediction. 

3. Add trusted goal storage for goal-spoofing attacks. 

4. Replace unrestricted recovery blending with gated recovery, including safe stop. 

## **10. Conclusion** 

TAIRO-HX keeps SAC+HER as the main robot policy but adds a smarter supervisory layer. The hierarchical classifier determines the current task stage, detects abnormal behavior, identifies the visible failure, estimates the likely attack family, and decides whether recovery is possible. 

The recovery manager then chooses whether to continue SAC+HER, correct its action, repair its observation, create a recovery subgoal, restore the trusted goal, slow down, or stop safely. 

The main idea is: 

**The robot should not apply the same type of recovery to every anomaly. It should first determine what is failing, whether trusted information remains, and whether recovery can be performed safely.** 

