# Container

`segbench.def` holds only the dependencies; the repository is mounted at run
time, so code and configuration changes need no rebuild.

```bash
./run.sh build                      # once: container/segbench.sif, with apptainer --fakeroot
./run.sh -c --quick                 # check, in the foreground
./run.sh -c -d --jobs 16            # full pipeline, detached
./run.sh -c -d scripts/03_noise.py --jobs 16   # one script, detached
```

## Detached runs

`-d` starts the run with `setsid nohup`: it gets its own session, ignores the
hang-up sent on logout, reads nothing from the terminal, and writes all output
to `logs/run_<YYYYmmdd_HHMMSS>.log`. The session id is saved in the matching
`.pid` file. `logs/latest.log` and `logs/latest.pid` point to the newest run.

```bash
tail -f logs/latest.log             # follow
./run.sh status                     # processes of the run and last log lines
./run.sh stop                       # stop the run and all its workers
```

The log starts with `=== started: <date> on <host>: <command>`, marks each
step with `=== NN_name`, and ends with `=== finished: <date>, exit status <code>`
(0 means every step succeeded) or `=== stopped: <date>`.
