# Key statistical findings — Recovery v4 HX variants

| finding | condition | delta | 95% CI | p_bh | significant |
| --- | --- | --- | --- | --- | --- |
| Plain v4 does significant HARM vs. no recovery | grip_state_falsification | -5.8% | [-8.2%, -3.3%] | 0.0002 | YES |
| v4-HX2 fixes the harm: significant win vs. plain v4 | grip_state_falsification | +4.2% | [+2.2%, +6.4%] | 0.0034 | YES |
| v4-HX2 vs. no recovery: parity restored (not significant either way) | grip_state_falsification | -1.6% | [-3.6%, +0.4%] | 0.8252 | no |
| v4-HX alone: object_pose_spoof regression, no longer significant at full grid | object_pose_spoof | -6.9% | [-12.4%, -1.6%] | 0.1802 | no |
| v4-HX3 targeted fix vs. plain v4: null result | object_pose_spoof | -2.2% | [-8.2%, +3.6%] | 1.0000 | no |
| v4-HX3 vs. adopted v4-HX2: null result | object_pose_spoof | +0.9% | [-4.9%, +6.4%] | 1.0000 | no |
| v4-HX3 preserves the hx2 grip_state_falsification win vs. plain v4 | grip_state_falsification | +5.1% | [+3.1%, +7.3%] | 3.4e-05 | YES |
