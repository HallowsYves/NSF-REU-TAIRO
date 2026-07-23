# Final TAIRO-HX comparison — all 8 mentor-requested metrics

(n=450/method, seeds 0-14, all 11 conditions, clean_2M checkpoint. Task-success/clean/safety are rates over all episodes; timing metrics are means across non-clean conditions/episodes where recovery triggered at least once; completion-time overhead is successful episodes only.)

| label | task_success_rate_overall | clean_task_performance | safety_violation_rate | detection_delay_steps | recovery_response_delay_steps | recovery_time_steps | num_interventions | completion_time_overhead_steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No recovery (SAC+HER) | 0.335 | 0.998 | 0.003 | — | — | — | — | — |
| Recovery v2 (earlier baseline) | 0.340 | 0.998 | 0.026 | 9.325 | 0.000 | 5.424 | 1.163 | 0.056 |
| Recovery v3 (earlier baseline) | 0.336 | 0.991 | 0.040 | 9.491 | 0.000 | 11.142 | 1.073 | 2.760 |
| Recovery v4 (gradual-response CCAR) | 0.334 | 0.998 | 0.001 | 15.909 | 76.097 | 100.309 | 0.856 | -1.647 |
| Recovery v4-HX6 (final, selective) | 0.335 | 1.000 | 0.001 | 20.025 | 66.517 | 95.494 | 1.174 | -0.398 |
