import wandb
try:
    api = wandb.Api()
    # Need to find the run ID or use the full path project/entity/run_id?
    # run = api.run("sebapusch-university-of-groningen/atari-lottery/atari_ddqn_lth_sweep-Pong-v5-S60-s2") works if that is the ID? No that is the name.
    # Must search.
    runs = api.runs(path="sebapusch-university-of-groningen/atari-lottery", filters={"display_name": "atari_ddqn_lth_sweep-Pong-v5-S60-s2"})
    if len(runs) > 0:
        run = runs[0]
        print(f"Run config: {run.config.get('pruning')}")
    else:
        print("Run not found")
except Exception as e:
    print(e)
