# Methods: grouped replay and replay diagnostics

We implement a FIFO replay buffer with capacity 50,000 and minibatches of 64
independent draws with replacement. Each transition carries an outage tag derived
only from the injected observation-delivery flags. A transition is tagged if its
current observation is lost, or if its next observation is lost and the transition
is nonterminal. Scenario timeouts end the finite task; for both terminal events
and timeouts, only the current delivery flag contributes. Target nondetection on
a delivered frame is distinct from an injected outage.

For a refreshed transition, we store a raw priority
\(r_i=|\delta_i|+10^{-6}\) and use mass \(u_i=r_i^{0.6}\). New transitions receive
the maximum raw priority in the current buffer immediately before FIFO eviction;
the first insertion into an empty buffer uses raw priority 1.0. Let \(N\) be the
current number of transitions and \(G_c\) the set with tag \(c\). Standard PER uses
\(P_{\mathrm{PER}}(i)=u_i/\sum_j u_j\). Grouped PER first samples a group with
probability \(|G_c|/N\), then samples a transition proportionally to its mass
within that group:

\[
P_{\mathrm{group}}(i)=\frac{|G_{c_i}|}{N}
\frac{u_i}{\sum_{j\in G_{c_i}}u_j}.
\]

Empty groups cannot be selected. Thus each group's total probability equals its
current buffer share in expectation, without imposing a fixed batch quota.
With one occupied group the method reduces to PER; with equal priorities it
reduces to uniform replay. The comparison samplers use
\(P_{\mathrm{uniform}}(i)=1/N\) and
\(P_{\mathrm{mix}}(i)=0.5/N+0.5P_{\mathrm{PER}}(i)\).

Every sampler returns its full marginal probability, including both mixture
branches. Importance weights are
\(w_i^{\mathrm{raw}}=(NP(i))^{-\beta}\), normalized by the maximum raw weight
over the entire current buffer. Equivalently, the implementation computes
\(w_i=(P_{\min}/P(i))^\beta\), where \(P_{\min}\) is the global minimum
probability. The exponent increases linearly from 0.4 to 1 over the shared
decision budget, including warm-up. The learner uses the mean of importance-
weighted, unreduced unit-threshold Huber losses. Priorities are refreshed after
the optimizer step using detached TD residuals from that loss computation.
Repeated draws contribute separately to the loss; repeated priority updates are
combined using their maximum absolute TD residual. Lifetime transition IDs
prevent updates to overwritten FIFO slots.

Non-finite TD residuals or losses abort the update before backpropagation, and
non-finite gradients abort it before the optimizer step. These checks do not clip
or replace residuals or gradients. Replay checkpoints retain transition payloads,
FIFO state, priorities, counts, and the replay generator state; they are separate
from analytical frozen-buffer exports and require the shared learner's other
states for complete training resumption.

Two group-specific sum/min trees and a global raw-priority max tree support
sampling, normalization, and changing priorities. Sampling takes
\(O(B\log C)\) time for batch size \(B\) and capacity \(C\), and metadata storage
is \(O(C)\). The same interface can be applied to the shared Double DQN learner
without changing its architecture, observations, rewards, or update budget.

Replay diagnostics record buffer occupancy at sampling time, outage draw counts,
TD residuals, per-transition replay counts, and each group's weighted Huber
numerator. We distinguish a group's within-group mean (denominator: group draws)
from its contribution to total minibatch loss (denominator: all draws). Across
updates, allocation is compared with the sum of minibatch size times the buffer
outage share at each sampling event, rather than with final buffer occupancy.
Sampler timings exclude the environment and learner; priority-update and logging
costs are benchmarked separately. Memory accounting reports ndarray metadata
storage and excludes observation payloads and Python overhead.

We additionally log the sampler's exact group marginal probability at each
sampling event. This distinguishes expected allocation from minibatch sampling
noise while retaining the observed allocation measures. Priority refresh batches
leaf updates and recomputes shared ancestors once per tree level. Frozen analysis
computes each global minimum probability once and reuses it for all exported
weights. Both optimizations preserve the original probability and weight formulas.

For a controlled sampler analysis, we export one frozen buffer's tags and raw
priorities and recompute all four probability vectors without changing its
contents. This isolates replay allocation on shared experience from differences
in data collected by separately trained agents. We validate the implementation
with exact probability/weight checks, ring-buffer and termination tests,
fixed-seed frequency tests, and CPU optimizer smoke tests. Synthetic examples
are labelled explicitly and do not establish effectiveness in ViZDoom.

The planned primary performance contrast is grouped PER minus PER hit rate after
training under 25% burst loss with mean length four decisions, evaluated on
held-out bursts of mean length eight. Performance, paired uncertainty estimates,
and evidence of PER's excess outage allocation must be assessed using the team's
actual training and evaluation runs. Replay frequency and weighted loss are
reported together; neither formula correctness nor synthetic allocation alone
supports a claim of improved policy performance.
