"""``plan``: mplib grasp planning in the planning environment."""


def main(argv=None):
    from soft_sim.planning.grasp import main as plan_main
    plan_main(argv)


if __name__ == "__main__":
    main()
