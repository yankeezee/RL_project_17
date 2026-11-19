# Does prioritized replay oversample observation outages in ViZDoom?

Anastasiia Ianke Areg Barseghyan Leon Zhu

## Course connection and research question

When target observations disappear, does TD-error prioritization spend too many updates on transitions involving missing information?

## Motivation, hypothesis and contribution

PER favors transitions with large prediction errors [1]. Recent UPER [2] and ReaPER [3] examine uncertainty and reliability as alternatives to raw TD error. We test the specific case of intermittent game-state observations.

**Hypothesis.** During burst loss, PER oversamples outage transitions relative to their buffer frequency, and preserving that frequency while prioritizing within groups improves held-out hit rate. Useful outage transitions might instead deserve extra replay, which would contradict the proposed mechanism. The contribution is a controlled test of a cheap, observable grouping rule and evidence about how to allocate a limited update budget.

## Task and controlled intervention

Use the built-in ViZDoom predict_position scenario [5], where a stationary player turns and fires one rocket at a moving target. Actions are no-op, turn left, turn right and fire, each held for four tics. Preserve the default kill reward, time penalty and episode limit. All agents use the same four-step history of target positions, heading, previous action, delivery/detection flags and observation age. Positions come from labels of visible targets on delivered frames. Missing positions are forward-filled with explicit masks. Engine velocity and hidden target positions are excluded from policy inputs.

Hold Double DQN [4], its network and training schedule fixed. Compare uniform replay, standard proportional PER, a 50/50 uniform–PER mixture, and grouped PER. Mark transition i with ci = 1 if an injected outage affects its current or nonterminal next observation. Let Gc contain transitions with tag c, N be buffer size and

ui = (|δi| + 10−6)0.6.

Sample by

PPER(i) = ui / Σj uj,

Pgroup(i) = |Gci| / N · ui / Σj∈Gci uj.

Thus grouped PER preserves each group’s buffer share in expectation. It still prioritizes within groups. Use each sampler’s actual probability for importance weights. The mixture tests whether a benefit can be explained by general reduction of prioritization.

## Experiment and evidence

Train each method under clean observations and under 25% burst loss with mean gap length four decisions. Evaluate frozen policies on clean observations, independent 25% loss, and 25% bursts of mean length four and eight. Use three training seeds and 100 held-out scenario/mask seeds shared across methods per test condition. Primary comparison is grouped PER minus PER hit rate after burst training, on eight-step bursts. Report seed-level differences and hierarchical bootstrap intervals, plus return, missed shots, timeouts, learning curves and wall-clock cost.

Measure the outage fraction in the buffer and replay batches, importance-weighted loss by group, and TD error versus replay count. Recompute sampler probabilities on a common frozen buffer to separate sampling from differences in collected experience. Support requires both a positive primary hit-rate difference and evidence of PER’s excess outage sampling. A benefit shared by the mixture weakens the case for outage-specific grouping. Confidence intervals spanning zero leave the performance claim unresolved.

## Budget, schedule and responsibilities

Cap the study at four samplers × two training regimes × three seeds × 300,000 decisions = 7.2 million decisions. Time a pilot before fixing any common reduction. Week 1 validates the environment and learner; week 2 runs training; week 3 evaluates and diagnoses replay; week 4 completes the paper, code and demo.

Anastasiia Ianke implements grouped replay, probability/weight checks and replay diagnostics. Areg Barseghyan implements Double DQN, uniform/PER baselines and training. Leon Zhu implements the environment, dropout process, mixture baseline and paired evaluation. All authors analyze results and present their contribution.

1

## Protocol details and research context

### Implementation choices fixed before the main experiment

**Observations.** Read world-plane target positions only from visible-object labels [6], normalized using fixed scenario bounds. Each history entry includes current player heading, previous action, delivered-frame flag, target-detected flag, last measured position, its age and an initialization flag. Before a first detection, use zero position and mark it uninitialized. A delivered frame with no target has a distinct encoding from an outage. Reset history each episode.

This is an idealized detector with a deliberately incomplete history. The grouping tag uses the injected delivery flags, which are already available to every learner. At a terminal transition, only the current flag contributes to the tag.

**Learner and update budget.** Use two 64-unit ReLU hidden layers, discount γ = 0.99, Adam with learning rate 10−4, Huber loss, a 50,000-transition FIFO buffer and batch size 64. Start updates after 5,000 collected decisions, then perform one optimizer step per four decisions. Copy the target network every 1,000 optimizer steps. Decay exploration probability linearly from 1 to 0.05 over the first 100,000 decisions. Initialize new priorities to the current buffer maximum and refresh sampled priorities after updates. Use the Double DQN target

yi = ri + γ(1 − zi)Q̄θ(s′i, arg maxa Qθ(s′i, a)),

δi = yi − Qθ(si, ai).

Here si denotes the fixed history representation and zi marks a task-ending transition. Treat the scenario’s timeout as the end of the finite task and include remaining time in every agent’s input. Count warm-up interactions in the budget and use the final checkpoint. Changes prompted by the pilot apply to all methods and are recorded before the main runs.

**Sampling correction.** For uniform replay P(i) = 1/N. For the mixture, P(i) = 0.5/N + 0.5PPER(i). For all samplers use wi = (NP(i))−β, with β increasing from 0.4 to 1 over training, normalized by the maximum weight over the current buffer. Empty groups have zero selection probability. Grouped PER becomes standard PER when all transitions have the same tag, providing a clean-condition implementation check. Sampling probabilities remain positive for every stored transition. Report both raw replay frequency and weighted loss because sampling frequency alone does not measure learning influence.

**Outage generation and reproducibility.** For stationary missing fraction p = 0.25 and mean burst length L, use recovery probability 1/L and outage-onset probability p/[L(1 − p)]. Start each episode with the channel available. Generate masks independently of actions and record realized missingness over each episode. Keep training, pilot and evaluation seeds disjoint. Shared evaluation seeds control the scenario initialization and observation channel, while policies may generate different trajectories. Publish code, pinned dependencies, scenario files, seeds, configs, learning curves and replay logs. Three training seeds limit the precision of uncertainty estimates.

## Why this is useful and how it relates to recent work

The practical decision is whether developers should spend additional updates on high-error experience collected during observation failures. The experiment can reveal when that choice improves learning and when it mainly repeats ambiguous experience. A useful output is a joint plot of replay allocation, sample efficiency and hit rate under held-out outage patterns. Grouping requires one stored flag per transition and two priority groups, making its overhead measurable.

UPER [2] targets learnable uncertainty, while ReaPER [3] adjusts priorities using TD-error reliability. These establish that raw surprise can be an imperfect learning signal. Our modest contribution is an empirical study using directly observable delivery status in one shooting task. The flag is a coarse proxy, and outage experience can contain valuable rewards and recovery information. Results will characterize that trade-off. A comparison with uniform replay and the mixture is essential to judge whether grouping adds value. Transfer beyond this scenario would require further experiments.

## References

[1] T. Schaul, J. Quan, I. Antonoglou and D. Silver. Prioritized Experience Replay. ICLR 2016.

[2] R. Carrasco-Davis, S. Lee, C. Clopath and W. Dabney. Uncertainty Prioritized Experience Replay. RLC 2025.

[3] L. S. Pleiss, T. Sutter and M. Schiffer. Reliability-Adjusted Prioritized Experience Replay. ICLR 2026. Published conference paper.

[4] H. van Hasselt, A. Guez and D. Silver. Deep Reinforcement Learning with Double Q-learning. AAAI 2016.

[5] Farama Foundation. ViZDoom default scenarios, Predict Position. Official documentation. Accessed 11 November 2026.

[6] Farama Foundation. ViZDoom GameState and Label API. Official documentation. Accessed 11 November 2026.

2
