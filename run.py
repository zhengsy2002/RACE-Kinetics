from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from race_kinetics.experiments import main

if __name__ == "__main__":
    main()
