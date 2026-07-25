<!-- Imported from `hunter_seeker_additional_components.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 7772b8a747695eeb45c3f6fdb179c1fc12e1c6901d498af5e8e557fa3ac4a705; original line count: 1075. -->

# Hunter Seeker Additional Components

## 1. Observation Learning Component

### Idea

Hunter Seeker should be able to learn from observed transitions, not only from actions it personally executes.

The core training unit is:

```text
frame_t → frame_t+1
```

When action labels are available, the training unit becomes:

```text
frame_t + action_t → frame_t+1
```

This allows the agent to learn from:

- its own recorded runs
- solver-generated trajectories
- failed/partial trajectories
- screen recordings
- later, external video footage

Observation learning should teach the agent:

- what changed
- which objects persisted across the transition
- which objects moved, appeared, disappeared, split, merged, or changed state
- which topology relations changed
- which transitions look like progress
- which transitions look dangerous or terminal
- which action probably caused the transition
- which click location probably caused a click-based transition

The first version should not require language reasoning. It should operate directly over frames, encoder features, topology features, object tables, and transition deltas.

The important distinction is:

```text
known-action transition:
    use for policy/world-model/action training

unlabeled observed transition:
    use for world-model/topology/event training
    use for policy training only if inverse-action confidence is high
```

Observation learning should improve Hunter Seeker by increasing candidate recall, actionability accuracy, click priors, topology trigger quality, and mechanism prediction.

---

### Implementation

#### 1.1 ObservationTransition

Create a canonical transition record.

```python
@dataclass
class ObservationTransition:
    frame_before: np.ndarray          # [64, 64]
    frame_after: np.ndarray           # [64, 64]

    action: Optional[int] = None       # known action if available
    click_x: int = -1                  # -1 if not click/unknown
    click_y: int = -1
    action_confidence: float = 0.0     # 1.0 for known labels, inferred otherwise

    changed_mask: Optional[np.ndarray] = None
    topology_delta: Optional[dict] = None
    object_delta: Optional[dict] = None
    event_type: Optional[str] = None

    terminal: bool = False
    progress_delta: float = 0.0
    source: str = "unknown"           # active_run, solver, video, etc.
```

Rules:

- Known trajectory labels get `action_confidence = 1.0`.
- Inferred labels from video get confidence from the inverse-action model.
- Do not use inferred actions for policy training unless confidence passes threshold.

Suggested threshold:

```python
INFERRED_ACTION_POLICY_THRESHOLD = 0.85
```

---

#### 1.2 ObservationReplayBuffer

Add a replay buffer for observation transitions.

Required sampling methods:

```python
class ObservationReplayBuffer:
    def push(self, transition: ObservationTransition): ...

    def sample_known_action_batch(self, batch_size): ...
    def sample_unlabeled_transition_batch(self, batch_size): ...
    def sample_click_transition_batch(self, batch_size): ...
    def sample_topology_delta_batch(self, batch_size): ...
    def sample_object_contrastive_batch(self, batch_size): ...
```

Data sources, in preferred order:

```text
1. Existing Hunter Seeker runs with known actions.
2. Solver-generated trajectories with known actions.
3. Failed/partial trajectories with known actions.
4. Screen recordings where actions can be inferred.
5. External video footage.
```

Reason: train inverse dynamics on known-action data first before trusting unlabeled video.

---

#### 1.3 EventSegmenter

Video should not be converted into a transition for every raw frame. Most frames are duplicates, flicker, or animation interpolation.

The event segmenter should convert raw video frames into stable before/after transitions:

```text
raw video frames
→ stable frame A
→ meaningful change begins
→ stable frame B
→ emit A → B
```

Initial simple implementation:

```python
def segment_video_frames(frames, min_change_fraction=0.002, stable_window=3):
    transitions = []
    last_stable = frames[0]

    for i in range(1, len(frames)):
        changed = frames[i] != last_stable
        change_fraction = changed.mean()

        if change_fraction < min_change_fraction:
            continue

        # wait for stabilization
        if i + stable_window < len(frames):
            future = frames[i:i + stable_window]
            stable = all((future[j] == future[-1]).mean() > 0.995 for j in range(len(future)))
            if stable:
                transitions.append((last_stable, future[-1]))
                last_stable = future[-1]

    return transitions
```

Later improvements:

- ignore visual flicker
- collapse animations
- detect object movement
- detect object split/merge/appearance/disappearance
- detect topology-level events
- retain intermediate animation frames only for future temporal modeling

---

#### 1.4 ChangedMaskHead

Train a head to localize which cells changed.

Input:

```text
frame_t
frame_t+1
encoder(frame_t)
encoder(frame_t+1)
latent_delta = z_t+1 - z_t
```

Target:

```python
changed_mask = (frame_before != frame_after)
```

Output:

```python
changed_mask_logits: [B, 64, 64]
```

Loss:

```python
changed_mask_loss = BCEWithLogitsLoss(pos_weight=...)
```

Use class weighting or focal loss, because most cells usually do not change.

Purpose:

- event localization
- click inference
- actionability learning
- distinguishing meaningful transition from noise
- supporting topology delta detection

---

#### 1.5 InverseActionModel

Train a model to infer which action caused a transition.

Training target:

```text
(frame_t, frame_t+1) → action_t
```

For click transitions:

```text
(frame_t, frame_t+1) → ACTION6 + click_x/click_y
```

Inputs:

```python
z_t = encoder(frame_t)
z_tp1 = encoder(frame_tp1)
delta = z_tp1 - z_t
changed_mask_features = changed_mask_head(...)
```

Outputs:

```python
action_logits       # [B, n_actions]
click_heatmap       # [B, 64, 64]
confidence_logit    # [B]
```

Training losses:

```python
action_loss = cross_entropy(action_logits, action)
click_loss = heatmap_or_coordinate_loss(click_heatmap, click_x, click_y)
confidence_loss = calibration_loss(confidence, action_correctness)
```

For known-action transitions:

```text
action label is ground truth
```

For video-only transitions:

```text
use predicted action only if confidence is high
otherwise use transition for world/topology/event learning only
```

Keyboard inference examples:

```text
object moved up    → ACTION1
object moved down  → ACTION2
object moved left  → ACTION3
object moved right → ACTION4
state changed in place → ACTION5 or ACTION6 depending on evidence
```

Click inference examples:

```text
localized changed region → likely click location
object toggled/selected/dragged → likely click center or causal object position
```

---

#### 1.6 TopologyDeltaHead

Train a head to describe structural changes between frames.

Inputs:

```text
object_table_t
object_table_t+1
topology_graph_t
topology_graph_t+1
latent_delta
changed_mask
```

Outputs should be multi-label event/topology classes:

```text
object_moved
object_appeared
object_disappeared
object_split
object_merged
color_changed
shape_changed
connectivity_changed
path_opened
path_blocked
goal_reached
hazard_contact
terminal_transition
no_meaningful_change
```

Loss:

```python
topology_delta_loss = BCEWithLogitsLoss(...)
```

Targets can initially be generated deterministically from existing object/topology extraction.

Purpose:

- teaches structural transition categories
- improves mechanism inference
- improves topology trigger logic
- prevents the system from learning only pixel-level dynamics

---

#### 1.7 Object Permanence / Contrastive Object Loss

Observation learning should teach stable object identity across frames.

For matched objects:

```text
same object across frame_t and frame_t+1 → embeddings close
```

For different objects:

```text
different objects → embeddings apart
```

Initial matching can use:

```text
IoU
overlap
centroid proximity
color
shape
topology signature
```

Loss options:

```text
contrastive loss
triplet loss
InfoNCE
```

Purpose:

- object permanence
- stable mechanism inference
- better movement tracking
- better observation from video
- better distinction between object identity and pixel position

---

#### 1.8 Forward Dynamics / Next-Frame Head

Extend the forward model to support both action-conditioned and observation-only training.

Action-conditioned mode:

```text
frame_t + action_t → frame_t+1
```

Observation-only mode:

```text
frame_t → likely change/event/next frame
```

Action-conditioned mode is more useful for policy.
Observation-only mode is useful for video learning.

Do not let pixel reconstruction dominate. The goal is causal/object/topology structure, not just visual reconstruction.

Suggested total loss:

```python
total_observation_loss = (
    0.5 * next_frame_loss
  + 1.0 * changed_mask_loss
  + 1.0 * inverse_action_loss_known
  + 0.2 * inverse_action_loss_inferred
  + 1.0 * click_loss
  + 1.5 * topology_delta_loss
  + 0.5 * object_contrastive_loss
)
```

---

#### 1.9 Encoder Drift Protection

Do not fully unfreeze the main encoder at first.

Use:

```text
frozen encoder
+ trainable observation adapter
+ trainable observation heads
```

Suggested adapter:

```python
class ObservationAdapter(nn.Module):
    def __init__(self, d_model=2048, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        return x + self.net(x)
```

Maintain an anchor encoder checkpoint and periodically compare:

```text
cosine(anchor_z, current_z)
norm_delta
topology probe health
object/actionability probe health
held-out candidate recall
```

Only selectively unfreeze encoder parts after observation heads prove useful without damaging existing behavior.

---

#### 1.10 Active-Play Integration

Observation learning should contribute extra priors during action selection.

Candidate score can receive optional bonuses:

```python
candidate_score = base_score
candidate_score += observation_actionability_bonus
candidate_score += inverse_dynamics_prior
candidate_score += topology_event_bonus
candidate_score += predicted_changed_mask_bonus
candidate_score += mechanism_familiarity_bonus
```

Hard rule:

```text
observation priors must not override terminal/hazard guards
```

Primary validation metric:

```text
Does the correct/useful action enter the candidate set more often after observation training?
```

Other metrics:

```text
known-action inverse accuracy
click heatmap top-k accuracy
click coordinate distance
changed-mask IoU
topology delta F1
object identity matching accuracy
next-frame loss
candidate-set recall lift
policy lift on held-out games
terminal/hazard regression check
```

---

## 2. Internalized Evaluator / Self-Model Component

### Idea

The evaluator should not remain only an external reranker that chooses after the agent has already proposed actions.

Bad pattern:

```text
policy proposes A
external evaluator prefers B
system executes B
```

Desired pattern:

```text
the agent internally scores B higher
policy chooses B itself
system executes B
```

The evaluator should become a teacher for an internal self-value model.

The internal self-model should learn:

- which candidate future it prefers
- which action it chose
- what it expected to happen
- whether the result matched expectation
- whether the choice was driven by progress, safety, topology, memory, or intrigue
- whether an external evaluator disagreed

This gives Hunter Seeker an internal value/arbitration signal instead of a permanent external authority.

Core principle:

```text
Do not wire the evaluator directly as final authority.
Use it to generate preference labels.
Train an internal SelfValueHead from those labels.
```

The self-value model should be trained relationally/pairwise whenever possible.

---

### Implementation

#### 2.1 CandidateFuture

Create a canonical candidate representation.

```python
@dataclass
class CandidateFuture:
    frame_t: np.ndarray

    action: int
    click_x: int = -1
    click_y: int = -1

    predicted_frame_tp1: Optional[np.ndarray] = None
    predicted_latent_tp1: Optional[torch.Tensor] = None
    predicted_changed_mask: Optional[torch.Tensor] = None

    topology_delta_features: Optional[torch.Tensor] = None
    object_delta_features: Optional[torch.Tensor] = None
    mechanism_features: Optional[torch.Tensor] = None

    progress_score: float = 0.0
    hazard_score: float = 0.0
    terminal_score: float = 0.0
    intrigue_score: float = 0.0
    memory_support: float = 0.0
    confidence: float = 0.0

    raw_score_components: Optional[dict] = None
```

All search/arbitration systems should eventually emit this structure instead of ad-hoc dictionaries.

Purpose:

- stable training data
- stable diagnostics
- easier pairwise comparison
- easier evaluator distillation
- easier self-model logging

---

#### 2.2 CandidateFutureEncoder

Encode candidate futures into a compact vector for self-value scoring.

Inputs may include:

```text
current latent state
candidate action embedding
click coordinate embedding
predicted next latent
predicted changed mask features
topology delta features
object delta features
mechanism features
progress/hazard/terminal/intrigue scalars
memory support
confidence
```

Sketch:

```python
class CandidateFutureEncoder(nn.Module):
    def __init__(self, d_latent=2048, d_model=512):
        super().__init__()
        self.action_embed = nn.Embedding(8, 64)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.scalar_mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.main = nn.Sequential(
            nn.LayerNorm(d_latent * 2 + 64 + 64 + 64),
            nn.Linear(d_latent * 2 + 64 + 64 + 64, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, state_latent, predicted_next_latent, action, coords, scalars):
        action_e = self.action_embed(action)
        coord_e = self.coord_mlp(coords)
        scalar_e = self.scalar_mlp(scalars)
        x = torch.cat([state_latent, predicted_next_latent, action_e, coord_e, scalar_e], dim=-1)
        return self.main(x)
```

The actual implementation should adapt to existing tensor shapes and score components.

---

#### 2.3 SelfValueHead

Pointwise internal value head:

```python
class SelfValueHead(nn.Module):
    def __init__(self, d_candidate=512):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(d_candidate),
            nn.Linear(d_candidate, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, candidate_features):
        return self.scorer(candidate_features).squeeze(-1)
```

This produces:

```text
higher score = internally preferred candidate future
```

It should eventually contribute to action selection, but only after shadow validation.

---

#### 2.4 PairwiseSelfPreferenceHead

Pairwise relational head for training and diagnostics.

```python
class PairwiseSelfPreferenceHead(nn.Module):
    def __init__(self, d_candidate=512):
        super().__init__()
        self.diff_norm = nn.LayerNorm(d_candidate, bias=False)
        self.scorer = nn.Sequential(
            nn.Linear(d_candidate, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, cand_a, cand_b):
        diff = self.diff_norm(cand_a - cand_b)
        return self.scorer(diff).squeeze(-1)
```

Positive score means:

```text
candidate A preferred over candidate B
```

Use random argument swaps during training to avoid order bias.

---

#### 2.5 SelfValue Training Data

Collect candidate sets during active play/search:

```python
@dataclass
class CandidateSetRecord:
    state_id: str
    frame_t: np.ndarray
    candidates: list[CandidateFuture]
    chosen_index: int
    executed_index: Optional[int]
    outcome: Optional[dict]
    source: str
```

Generate pairwise labels from multiple teachers.

##### Outcome teacher

```text
progress candidate > no-progress candidate
non-terminal candidate > terminal/death candidate
successful candidate > failed candidate
level-completing candidate > non-completing candidate
```

##### Evaluator teacher

Use external evaluator or trajectory evaluator to compare candidate futures:

```text
candidate_i preferred over candidate_j
```

Store these as labels, not as final action overrides.

##### Safety teacher

Hard preference labels:

```text
safe candidate > death candidate
safe candidate > unnecessary hazard candidate
progress-safe candidate > high-risk uncertain candidate
```

##### Observation teacher

From the observation component:

```text
candidate matching watched-successful transition > implausible alternative
candidate matching watched-hazard transition < safe alternative
```

---

#### 2.6 Pairwise Loss

For pointwise SelfValueHead:

```python
score_a = self_value(features_a)
score_b = self_value(features_b)
loss = -F.logsigmoid(target * (score_a - score_b)).mean()
```

For PairwiseSelfPreferenceHead:

```python
pair_score = pairwise_head(features_a, features_b)
loss = -F.logsigmoid(target * pair_score).mean()
```

Where:

```text
target = +1 if A preferred over B
target = -1 if B preferred over A
```

Regularize score magnitude:

```python
loss += 1e-4 * score.pow(2).mean()
```

Use random candidate order swaps.

---

#### 2.7 Auxiliary Self-Model Heads

The self-model should not only output value. It should also predict its own decision and expected outcome.

Auxiliary heads:

```text
chosen_action_head:
    self_state_t → chosen action

expected_change_head:
    state + action → will frame change?

expected_progress_head:
    state + action → progress estimate

expected_terminal_head:
    state + action → terminal/death risk

confidence_head:
    state + action → probability prediction is reliable

surprise_head:
    expected outcome vs actual outcome mismatch
```

Purpose:

- makes decisions interpretable
- helps debugging
- prevents self-value from becoming an opaque scalar
- lets the system learn when it was wrong

---

#### 2.8 SelfState Logging

Store a self-state per step.

```python
@dataclass
class SelfState:
    current_latent: Optional[torch.Tensor]
    previous_latent: Optional[torch.Tensor]

    chosen_action: int
    click_x: int = -1
    click_y: int = -1

    predicted_outcome: Optional[dict] = None
    actual_outcome: Optional[dict] = None

    self_value_score: float = 0.0
    confidence: float = 0.0
    hazard_estimate: float = 0.0
    progress_estimate: float = 0.0
    intrigue_estimate: float = 0.0
    memory_support: float = 0.0

    evaluator_score: Optional[float] = None
    evaluator_disagreement: bool = False
    surprise: float = 0.0
```

Diagnostics should answer:

```text
What did I choose?
What did I expect?
What actually happened?
Was I surprised?
Was there evaluator disagreement?
Was the action chosen because of progress, safety, topology, memory, or intrigue?
```

---

#### 2.9 Inference Integration

Do not immediately let SelfValueHead control action selection.

Staged rollout:

```text
Phase A: train offline only
Phase B: shadow mode, log scores but do not affect actions
Phase C: small score contribution
Phase D: tie-break / uncertainty-only contribution
Phase E: normal contribution after validation
```

Suggested weights:

```python
self_value_weight = 0.0   # offline/shadow
self_value_weight = 0.1   # early live test
self_value_weight = 0.25  # after calibration
```

Candidate scoring:

```python
score = base_score
score += self_value_weight * self_value_score
```

Hard rule:

```text
SelfValueHead must not override terminal/death/hazard hard guards.
```

---

#### 2.10 External Evaluator Role

The evaluator should move through these roles:

```text
1. Teacher:
   generates pairwise labels over candidate futures.

2. Critic:
   audits decisions and flags disagreement.

3. Uncertainty tool:
   invoked only when internal scores are close or confidence is low.

4. Retired from hot path:
   internal SelfValueHead carries the value signal during normal play.
```

The evaluator should not remain a permanent external final authority unless experiments prove it is necessary.

---

#### 2.11 Anti-Degeneration Diagnostics

Use diagnostics similar to pairwise evaluator flip tests.

Required checks:

```text
A vs B score
B vs A score
antisymmetry correlation
sign flip rate
constant-score detection
score magnitude distribution
candidate-order bias
terminal preference check
hazard preference check
progress preference check
```

Watch for failures:

```text
always prefers first candidate
always outputs positive
always prefers high-intrigue noise
always prefers progress even when terminal
copies old topology score without learning outcome
overfits to candidate order
score magnitude explosion
```

Training safeguards:

```text
random candidate order swaps
score L2 regularization
biasless LayerNorm on pairwise difference path
held-out candidate pair validation
terminal/hazard regression tests
```

---

#### 2.12 Validation Metrics

Track:

```text
pairwise preference accuracy on held-out candidate pairs
progress-vs-no-progress preference accuracy
terminal/death avoidance preference accuracy
hazard avoidance preference accuracy
agreement with external evaluator
agreement with actual outcomes
confidence calibration
flip-test antisymmetry correlation
candidate-order bias
policy lift when enabled
regression against baseline action selector
```

Most important final metric:

```text
Does enabling SelfValueHead improve held-out game performance without reducing safety or terminal avoidance?
```

---

## Combined Integration

Observation learning and self-value should eventually connect.

Observation learning provides:

```text
this transition pattern was seen before
this action likely caused it
this topology/event pattern tends to mean progress
this topology/event pattern tends to mean hazard
```

Self-value uses that information:

```text
prefer candidate matching known successful transition
avoid candidate matching known hazardous transition
prefer candidate with familiar causal mechanism
avoid candidate with high predicted terminal risk
```

Active loop:

```text
frame
→ encode frame
→ extract object/topology/event state
→ generate candidate actions
→ predict candidate futures
→ enrich candidates with observation-learned priors
→ score candidates with internal SelfValueHead
→ choose action
→ execute action
→ log expected vs actual outcome
→ update observation buffer and self-value training data
```

Implementation order:

```text
1. Add ObservationTransition and ObservationReplayBuffer.
2. Feed known-action trajectories into observation buffer.
3. Train InverseActionModel on known trajectories.
4. Add ChangedMaskHead.
5. Add TopologyDeltaHead.
6. Add video EventSegmenter.
7. Use high-confidence inferred labels from video.
8. Define CandidateFuture.
9. Train SelfValueHead in shadow mode.
10. Add SelfState logging.
11. Add small self-value contribution to scoring.
12. Move external evaluator toward teacher/audit role.
```
