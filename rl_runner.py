from data_loader import load_raw_close


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
