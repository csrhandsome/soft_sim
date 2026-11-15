"""``plan``: grasp planning (defaults to cuRobo when available, else mplib)."""


def main(argv=None):
    try:
        import curobo  # noqa: F401
    except ImportError:
        from soft_sim.planning.grasp import main as plan_main
    else:
        from soft_sim.planning.curobo import main as plan_main
    plan_main(argv)


if __name__ == "__main__":
    main()
