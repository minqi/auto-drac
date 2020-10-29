import datetime
import submitit
import os
import sys
from coolname import generate_slug

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("path", "~/cmds.txt", "Path to list of commands to run.")
flags.DEFINE_string("name", "anyslurm", "Experiment name.")
flags.DEFINE_boolean("debug", False, "Only debugging output.")


def arg2str(k, v):
    if isinstance(v, bool):
        if v:
            return ("--%s" % k,)
        else:
            return ""
    else:
        return ("--%s" % k, str(v))


def launch_experiment_and_remotenv(experiment_args):
    # imports and definition are inside of function because of submitit
    import multiprocessing as mp

    def launch_experiment(experiment_args):
        import subprocess
        import itertools

        python_exec = sys.executable
        args = itertools.chain(*[arg2str(k, v) for k, v in experiment_args.items()])
        subprocess.call([python_exec, "-m", "train"] + list(args))

    experiment_process = mp.Process(target=launch_experiment, args=[experiment_args])
    experiment_process.start()
    experiment_process.join()


def main(argv):
    now = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")

    rootdir = os.path.expanduser(f"~/{FLAGS.name}")
    submitit_dir = os.path.expanduser(f"~/{FLAGS.name}/{now}")
    executor = submitit.SlurmExecutor(folder=submitit_dir)
    os.makedirs(submitit_dir, exist_ok=True)

    symlink = os.path.join(rootdir, "latest")
    if os.path.islink(symlink):
        os.remove(symlink)
    if not os.path.exists(symlink):
        os.symlink(submitit_dir, symlink)
        print("Symlinked experiment directory: %s", symlink)

    all_args = list()

    with open(os.path.expanduser(FLAGS.path), "r") as f:
        cmds = "".join(f.readlines()).split("\n\n")
        cmds = [cmd.split("\\\n")[1:] for cmd in cmds]
        cmds = [cmd for cmd in cmds if len(cmd) > 0]
        for line in cmds:
            le_args = dict()
            # print(line)
            for pair in line:
                key, val = pair.strip()[2:].split("=")
                le_args[key] = val
            if "run_name" not in le_args:
                le_args["run_name"] = generate_slug()

            all_args.append(le_args)

    executor.update_parameters(
        # examples setup
        partition="learnfair",
        # partition="priority",
        comment="ICLR 2021 submission",
        # time=1 * 24 * 60,
        time=1 * 8 * 60,
        nodes=1,
        ntasks_per_node=1,
        # job setup
        job_name=FLAGS.name,
        mem="60GB",
        cpus_per_task=20,
        num_gpus=1,
        # constraint="volta32gb",
        array_parallelism=100,
    )

    print("\nAbout to submit", len(all_args), "jobs")

    if not FLAGS.debug:
        job = executor.map_array(launch_experiment_and_remotenv, all_args)

        for j in job:
            print("Submitted with job id: ", j.job_id)
            print(f"stdout -> {submitit_dir}/{j.job_id}_0_log.out")
            print(f"stderr -> {submitit_dir}/{j.job_id}_0_log.err")

        print(f"Submitted {len(job)} jobs!")

        print()
        print(submitit_dir)


if __name__ == "__main__":
    app.run(main)
