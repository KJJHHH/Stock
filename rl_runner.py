from data_loader import load_raw_close
import os
import numpy as np


def run_rl(ticker, train_start, train_end, backtest_start, backtest_end, config, device, transaction_cost):
    from rl.rl_env import TradingEnv
    from rl.rl_train import (
        evaluate_policy,
        make_actor_policy,
        make_continuous_policy,
        make_dqn_policy,
        train_a2c,
        train_dqn,
        train_ppo,
        train_sac,
        train_td3,
    )

    rl_config = config.get("rl", {})
    rl_algorithm = rl_config.get("algorithm", "sac")  # Options: dqn, ppo, a2c, ppo_tuned, sac, td3
    rl_window = rl_config.get("window", 100)

    train_prices, _ = load_raw_close(ticker, train_start, train_end)
    backtest_prices, _ = load_raw_close(ticker, backtest_start, backtest_end)

    continuous_action = rl_algorithm in {"sac", "td3"}
    train_env = TradingEnv(
        train_prices,
        window_size=rl_window,
        transaction_cost=transaction_cost,
        continuous=continuous_action,
    )
    backtest_env = TradingEnv(
        backtest_prices,
        window_size=rl_window,
        transaction_cost=transaction_cost,
        continuous=continuous_action,
    )

    if rl_algorithm == "dqn":
        dqn_config = rl_config.get("dqn", {})
        model = train_dqn(train_env, dqn_config, device)
        policy = make_dqn_policy(model, device)
    elif rl_algorithm == "ppo":
        ppo_config = rl_config.get("ppo", {})
        model = train_ppo(train_env, ppo_config, device)
        policy = make_actor_policy(model, device)
    elif rl_algorithm == "ppo_tuned":
        ppo_config = rl_config.get("ppo_tuned", {})
        model = train_ppo(train_env, ppo_config, device)
        policy = make_actor_policy(model, device)
    elif rl_algorithm == "a2c":
        a2c_config = rl_config.get("a2c", {})
        model = train_a2c(train_env, a2c_config, device)
        policy = make_actor_policy(model, device)
    elif rl_algorithm == "sac":
        sac_config = rl_config.get("sac", {})
        model = train_sac(train_env, sac_config, device)
        policy = make_continuous_policy(model, device)
    elif rl_algorithm == "td3":
        td3_config = rl_config.get("td3", {})
        model = train_td3(train_env, td3_config, device)
        policy = make_continuous_policy(model, device)
    else:
        raise ValueError("Invalid RL algorithm")

    equities = evaluate_policy(backtest_env, policy, episodes=1)
    print(f"RL {rl_algorithm} backtest equity: {equities[-1]:.2f}")

    model_name = f"RL_{rl_algorithm.upper()}"
    model_dir = os.path.join("result", f"{model_name}-result")
    os.makedirs(model_dir, exist_ok=True)
    plot_name = f"backtest_{ticker.replace('^', '')}_{model_name}.png"
    plot_path = os.path.join(model_dir, plot_name)

    initial_asset = 100.0
    equity_curve = np.asarray(equities, dtype=np.float64) * initial_asset
    buy_hold = initial_asset * np.cumprod(1 + backtest_env.returns[backtest_env.window_size :], axis=0)
    min_len = min(len(equity_curve), len(buy_hold))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve[:min_len], label=f"{model_name} strategy equity")
    plt.plot(buy_hold[:min_len], label="Buy and hold equity")
    plt.xlabel("Step")
    plt.ylabel("Asset Value")
    plt.title(f"{ticker} Backtest Equity ({model_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved backtest plot to {plot_path}")
