import collections
import re
import warnings

import numpy as np

import dreamerv2 as dm2


def train(agent, env, train_replay, eval_replay, logger, args):
    logdir = dm2.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_train = dm2.when.Every(args.train_every)
    should_log = dm2.when.Every(args.log_every)
    should_expl = dm2.when.Until(args.expl_until)
    should_video = dm2.when.Every(args.eval_every)
    step = logger.step

    timer = dm2.Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    if hasattr(train_replay, "_sample"):
        timer.wrap("replay", train_replay, ["_sample"])

    nonzeros = set()

    def per_episode(ep):
        metrics = {}
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        print(f"Episode has {length} steps and return {score:.1f}.")
        metrics["length"] = length
        metrics["score"] = score
        metrics["reward_rate"] = (ep["reward"] - ep["reward"].min() >= 0.1).mean()
        logs = {}
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                logs[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                logs[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                logs[f"max_{key}"] = ep[key].max(0).mean()
        if should_video(step):
            for key in args.log_keys_video:
                if key == "none":
                    continue
                metrics[f"policy_{key}"] = ep[key]
                if f"log_goal_{key}" in ep:
                    if ep[key].shape == ep[f"log_goal_{key}"].shape:
                        goal = (255 * ep[f"log_goal_{key}"]).astype(np.uint8)
                        metrics[f"policy_{key}_with_goal"] = np.concatenate([ep[key], goal], 2)
        logger.add(metrics, prefix="episode")
        logger.add(logs, prefix="logs")
        logger.add(train_replay.stats, prefix="replay")
        logger.write()

    def create_agent():
        random_agent = dm2.RandomAgent(env.act_space)
        return random_agent

    fill = max(0, args.eval_fill - len(eval_replay))
    if fill:
        print(f"Fill eval dataset ({fill} steps).")
        eval_driver = dm2.Driver(env)
        eval_driver.on_step(lambda ep, info, worker: eval_replay.add(ep, worker))
        random_agent = create_agent()
        eval_driver(random_agent.policy, steps=fill, episodes=1)
        del eval_driver

    driver = dm2.Driver(env)
    driver.on_episode(lambda ep, info, worker: per_episode(ep))
    driver.on_step(lambda ep, info, worker: step.increment())
    driver.on_step(lambda ep, info, worker: train_replay.add(ep, worker))
    fill = max(0, args.train_fill - len(train_replay))
    if fill:
        print(f"Fill train dataset ({fill} steps).")
        random_agent = create_agent()
        driver(random_agent.policy, steps=fill, episodes=1)

    dataset_train = iter(agent.dataset(train_replay.dataset))
    dataset_eval = iter(agent.dataset(eval_replay.dataset))
    state = [None]  # To be writable from train step function below.
    assert args.pretrain > 0  # At least one step to initialize variables.
    for _ in range(args.pretrain):
        _, state[0], _ = agent.train(next(dataset_train), state[0])

    metrics = collections.defaultdict(list)
    batch = [None]

    def train_step(tran, info, worker):
        if should_train(step):
            for _ in range(args.train_steps):
                batch[0] = next(dataset_train)
                outs, state[0], mets = agent.train(batch[0], state[0])
                [metrics[key].append(value) for key, value in mets.items()]
                if "priority" in outs:
                    train_replay.prioritize(outs["key"], outs["priority"])
        if should_log(step):
            with warnings.catch_warnings():  # Ignore empty slice warnings.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for name, values in metrics.items():
                    logger.scalar("train/" + name, np.nanmean(values, dtype=np.float64))
                    metrics[name].clear()
            logger.add(agent.report(batch[0]), prefix="report")
            logger.add(agent.report(next(dataset_eval)), prefix="eval")
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)

    driver.on_step(train_step)

    checkpoint = dm2.Checkpoint(logdir / "checkpoint.pkl")
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.train_replay = train_replay
    checkpoint.eval_replay = eval_replay
    checkpoint.load_or_save()

    print("Start training loop.")
    policy = lambda *args: agent.policy(*args, mode="explore" if should_expl(step) else "train")
    while step < args.steps:
        logger.write()
        driver(policy, steps=args.eval_every)
        checkpoint.save()
