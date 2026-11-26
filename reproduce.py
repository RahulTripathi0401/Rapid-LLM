"""One-shot reproduction of every Project 3 (block diffusion) result in the presentation and report.

After the setup in README.md:

    python reproduce.py            # everything, about 50 min on 8 cores
    python reproduce.py --quick    # about 5 min: skips the two long sweeps and replots from the saved CSVs

Steps: unit tests, the feature-off regression check, every evaluation run, the tiler probe,
the report and slide figures, and course_p3/results/presentation_numbers.md, which lists every
number on every slide with the file it comes from. Outputs are listed at the end.
"""

import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.join(REPO, "course_p3")


def step(title, args):
    print(f"\n=== {title}\n$ {' '.join(args)}", flush=True)
    start = time.time()
    subprocess.run([sys.executable, *args], cwd=REPO, check=True)
    print(f"--- done in {time.time() - start:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="skip the prompt-length sweep and the A100/H100 crossover (about 45 min); "
                         "their figures and tables are rebuilt from the CSVs in course_p3/results/")
    args = ap.parse_args()
    begin = time.time()

    step("Unit tests", ["-m", "pytest", "-q"])
    step("Feature off: 11 runs must match the original code byte for byte", [os.path.join(P3, "regress.py"), "check"])
    step("Required evaluation (presentation slide 6)", [os.path.join(P3, "evaluate.py"), "required"])
    if not args.quick:
        step("Case study 1: prompt length x batch x (b, R) (slide 8)",
             [os.path.join(P3, "evaluate.py"), "sweep"])
        step("Case study 2: A100 vs H100 crossover (slides 3, 7, 9, 10)",
             [os.path.join(P3, "evaluate.py"), "crossover"])
    step("Output-head sensitivity (slide 11)", [os.path.join(P3, "evaluate.py"), "head"])
    step("Tiler cliff probe (slide 11)", [os.path.join(P3, "tiler_probe.py")])
    step("Report figures", [os.path.join(P3, "plots.py")])
    step("Presentation figures", [os.path.join(P3, "slide_figs.py")])
    step("Every number on every slide", [os.path.join(P3, "summarize.py")])

    note = "  (--quick: sweep.csv and crossover.csv were not re-run)" if args.quick else ""
    print(f"""
All done in {(time.time() - begin) / 60:.1f} min.{note}

Outputs (see "Reproducing the presentation" in README.md for the slide-by-slide map):
  course_p3/results/presentation_numbers.md   every number on every slide, with its source CSV
  course_p3/results/*.csv                      raw results of every run
  course_p3/figures/slides/*.png               the three presentation figures
  course_p3/figures/*.png                      the report figures
  course_p3/REPORT.md                          the written report
""")


if __name__ == "__main__":
    main()
