
import argparse
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F

from env import PredictPositionEnv, make_outage, train_episode_seed, CURVE_SEED_BASE
from networks import mlp, as_tensor, dqn_target
from replay import ReplayBuffer


@torch.no_grad()
def evaluate_greedy(
    q_net,
    condition="clean",
    n_episodes=100,
    seed=CURVE_SEED_BASE
):
    env = PredictPositionEnv(
        outage=make_outage(condition)
    )

    returns = []
    hits = []

    q_net.eval()

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        episode_return = 0.0

        while not done:
            action = int(
                q_net(as_tensor(obs).unsqueeze(0))
                .argmax(dim=1).item()
            )

            obs, reward, done, info = env.step(action)
            episode_return += reward

        returns.append(episode_return)
        hits.append(info["hit"])

    env.close()
    q_net.train()

    return float(np.mean(returns)), float(np.mean(hits))


def safe_mean(values):
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def train(
    seed=0,
    sampler="uniform",
    condition="clean",
    total_steps=300_000,
    gamma=0.99,
    lr=1e-4,
    batch_size=64,
    buffer_capacity=50_000,
    learning_starts=5_000,
    train_every=4,
    target_sync_every=1_000,
    log_every=1_000,
):
    start_time = time.perf_counter()

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Environment
    env = PredictPositionEnv(
        outage=make_outage(condition)
    )

    # Online and target networks
    q_net = mlp(env.obs_size, env.n_actions)
    target_net = mlp(env.obs_size, env.n_actions)

    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(
        q_net.parameters(), lr=lr
    )

    # Replay buffer
    buffer = ReplayBuffer(
        capacity=buffer_capacity,
        obs_dim=env.obs_size,
        sampler=sampler
    )

    episode_index = 0
    obs = env.reset(seed=train_episode_seed(seed, episode_index))

    optimizer_steps = 0
    episode_return = 0.0

    returns = []
    hits = []
    timeouts = []

    # Replay diagnostics
    buffer_outage_history = []
    sample_outage_history = []
    expected_outage_history = []

    loss_history = []
    td_error_history = []
    q_history = []

    outage_td_history = []
    clean_td_history = []

    outage_loss_history = []
    clean_loss_history = []

    records = []

    run_name = f"ddqn_{sampler}_{condition}_seed{seed}"

    # Training loop
    for step in range(1, total_steps + 1):

        # Exploration schedule
        epsilon = 1.0 - 0.95 * min(
            step / 100_000, 1.0
        )

        # Importance-sampling correction schedule
        beta = 0.4 + 0.6 * step / total_steps

        # Epsilon-greedy action selection
        if rng.random() < epsilon:
            action = int(rng.integers(env.n_actions))
        else:
            with torch.no_grad():
                action = int(
                    q_net(as_tensor(obs).unsqueeze(0))
                    .argmax(dim=1).item()
                )

        # Collect transition
        current_delivered = env.delivered

        next_obs, reward, done, info = env.step(action)

        # An outage affects either the current observation
        # or the nonterminal next observation.
        outage = (
            not current_delivered
            or (not done and not info["delivered"])
        )

        buffer.push(
            obs,
            action,
            reward,
            next_obs,
            done,
            outage
        )

        episode_return += reward

        # Episode termination
        if done:
            returns.append(episode_return)
            hits.append(info["hit"])
            timeouts.append(info["timeout"])

            episode_return = 0.0

            episode_index += 1
            obs = env.reset(
                seed=train_episode_seed(seed, episode_index)
            )
        else:
            obs = next_obs

        # Double DQN optimization
        if (
            step >= learning_starts
            and step % train_every == 0
            and len(buffer) >= batch_size
        ):
            samples = buffer.sample(
                batch_size, rng, beta
            )

            # ------------------------------------------
            # Replay sampling diagnostics
            # ------------------------------------------

            buffer_outages = buffer.outages[:buffer.size]
            sampled_outages = samples["outages"]

            buffer_fraction = buffer_outages.mean()
            sample_fraction = sampled_outages.mean()

            # Expected sampling fraction under the
            # current replay probability distribution.
            probs = buffer.probabilities()

            expected_fraction = probs[
                buffer_outages
            ].sum()

            buffer_outage_history.append(
                float(buffer_fraction)
            )

            sample_outage_history.append(
                float(sample_fraction)
            )

            expected_outage_history.append(
                float(expected_fraction)
            )

            # ------------------------------------------
            # Convert samples to tensors
            # ------------------------------------------

            batch = {
                k: as_tensor(
                    samples[k],
                    torch.long if k == "actions"
                    else torch.float32
                )
                for k in [
                    "obs",
                    "actions",
                    "rewards",
                    "next_obs",
                    "dones"
                ]
            }

            weights = as_tensor(samples["weights"])

            # Double DQN regression target
            target = dqn_target(
                batch,
                q_net,
                target_net,
                gamma
            )

            # Predicted Q-values for selected actions
            q_pred = q_net(batch["obs"]).gather(
                1,
                batch["actions"].view(-1, 1)
            ).squeeze(1)

            # TD errors
            td_errors = target - q_pred

            # Importance-weighted Huber loss
            per_sample_loss = F.smooth_l1_loss(
                q_pred,
                target,
                reduction="none"
            )

            weighted_losses = weights * per_sample_loss

            loss = weighted_losses.mean()

            # Optimization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            optimizer_steps += 1

            # ------------------------------------------
            # Update replay priorities
            # ------------------------------------------

            abs_td = (
                td_errors.detach()
                .abs()
                .cpu()
                .numpy()
            )

            if sampler != "uniform":
                buffer.update_priorities(
                    samples["indices"],
                    abs_td
                )

            # ------------------------------------------
            # Learning diagnostics
            # ------------------------------------------

            loss_history.append(float(loss.item()))

            td_error_history.append(
                float(abs_td.mean())
            )

            q_history.append(
                float(q_pred.detach().mean().item())
            )

            weighted_loss_np = (
                weighted_losses.detach()
                .cpu()
                .numpy()
            )

            outage_mask = sampled_outages
            clean_mask = ~sampled_outages

            outage_td_history.append(
                safe_mean(abs_td[outage_mask])
            )

            clean_td_history.append(
                safe_mean(abs_td[clean_mask])
            )

            # Contribution of each group to the
            # total importance-weighted batch loss.
            outage_loss_history.append(
                float(
                    weighted_loss_np[outage_mask].sum()
                    / batch_size
                )
            )

            clean_loss_history.append(
                float(
                    weighted_loss_np[clean_mask].sum()
                    / batch_size
                )
            )

            # Target network synchronization
            if optimizer_steps % target_sync_every == 0:
                target_net.load_state_dict(
                    q_net.state_dict()
                )

        # ----------------------------------------------
        # Logging
        # ----------------------------------------------

        if step % log_every == 0:

            mean_return = safe_mean(returns[-100:])
            hit_rate = safe_mean(hits[-100:])
            timeout_rate = safe_mean(timeouts[-100:])

            window = 250

            bf = safe_mean(
                buffer_outage_history[-window:]
            )

            sf = safe_mean(
                sample_outage_history[-window:]
            )

            ef = safe_mean(
                expected_outage_history[-window:]
            )

            oversampling = sf / bf if bf > 0 else np.nan

            expected_oversampling = (
                ef / bf if bf > 0 else np.nan
            )

            mean_loss = safe_mean(
                loss_history[-window:]
            )

            mean_td = safe_mean(
                td_error_history[-window:]
            )

            mean_q = safe_mean(
                q_history[-window:]
            )

            outage_td = safe_mean([
                x for x in outage_td_history[-window:]
                if np.isfinite(x)
            ])

            clean_td = safe_mean([
                x for x in clean_td_history[-window:]
                if np.isfinite(x)
            ])

            outage_loss = safe_mean(
                outage_loss_history[-window:]
            )

            clean_loss = safe_mean(
                clean_loss_history[-window:]
            )

            elapsed = time.perf_counter() - start_time

            print(
                f"Step {step:7d} | "
                f"Return {mean_return:+.3f} | "
                f"Hit rate {hit_rate:.3f} | "
                f"Epsilon {epsilon:.3f}"
            )

            if loss_history:
                print(
                    f"  Buffer outage: {bf:.3f} | "
                    f"Sample outage: {sf:.3f} | "
                    f"Expected outage: {ef:.3f} | "
                    f"Oversampling: {oversampling:.3f}"
                )

                print(
                    f"  Loss: {mean_loss:.6f} | "
                    f"TD error: {mean_td:.4f} | "
                    f"Mean Q: {mean_q:.4f}"
                )

                print(
                    f"  Outage TD: {outage_td:.4f} | "
                    f"Clean TD: {clean_td:.4f} | "
                    f"Outage loss: {outage_loss:.6f} | "
                    f"Clean loss: {clean_loss:.6f}"
                )

            records.append({
                "step": step,
                "optimizer_steps": optimizer_steps,
                "seed": seed,
                "sampler": sampler,
                "condition": condition,
                "epsilon": epsilon,
                "beta": beta,
                "train_return": mean_return,
                "train_hit_rate": hit_rate,
                "timeout_rate": timeout_rate,
                "buffer_size": len(buffer),
                "buffer_outage_fraction": bf,
                "sample_outage_fraction": sf,
                "expected_outage_fraction": ef,
                "oversampling_ratio": oversampling,
                "expected_oversampling_ratio":
                    expected_oversampling,
                "loss": mean_loss,
                "mean_abs_td_error": mean_td,
                "mean_q": mean_q,
                "outage_td_error": outage_td,
                "clean_td_error": clean_td,
                "outage_weighted_loss": outage_loss,
                "clean_weighted_loss": clean_loss,
                "elapsed_seconds": elapsed,
            })

    env.close()

    training_time = time.perf_counter() - start_time

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    torch.save(
        q_net.state_dict(),
        f"{run_name}.pt"
    )

    # Save training diagnostics
    if records:
        with open(
            f"{run_name}_logs.csv",
            "w",
            newline=""
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=records[0].keys()
            )

            writer.writeheader()
            writer.writerows(records)

    # --------------------------------------------------
    # Final greedy evaluation
    # --------------------------------------------------

    eval_return, eval_hit_rate = evaluate_greedy(
        q_net,
        condition=condition,
        n_episodes=100,
        seed=CURVE_SEED_BASE
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"Sampler: {sampler}")
    print(f"Condition: {condition}")
    print(f"Seed: {seed}")
    print(f"Training steps: {total_steps}")
    print(f"Optimizer steps: {optimizer_steps}")
    print(f"Greedy return: {eval_return:.3f}")
    print(f"Greedy hit rate: {eval_hit_rate:.3f}")
    print(f"Training time: {training_time:.2f} seconds")

    print(f"Model saved to: {run_name}.pt")
    print(f"Logs saved to: {run_name}_logs.csv")

    # Save final results
    summary = {
        "sampler": sampler,
        "condition": condition,
        "seed": seed,
        "training_steps": total_steps,
        "optimizer_steps": optimizer_steps,
        "greedy_return": eval_return,
        "greedy_hit_rate": eval_hit_rate,
        "training_seconds": training_time,
    }

    with open(
        f"{run_name}_summary.csv",
        "w",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary.keys()
        )

        writer.writeheader()
        writer.writerow(summary)

    return q_net


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sampler",
        choices=["uniform", "per"],
        default="uniform"
    )

    parser.add_argument(
        "--condition",
        choices=[
            "clean",
            "indep25",
            "burst25_L4",
            "burst25_L8"
        ],
        default="clean"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=300_000
    )

    args = parser.parse_args()

    train(
        seed=args.seed,
        sampler=args.sampler,
        condition=args.condition,
        total_steps=args.steps
    )
