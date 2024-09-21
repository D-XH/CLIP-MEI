import sys
sys.path.append('.')
from utils.config import Config
from main_run import Learner

def main():
    cfg = Config(load=True)
    # print(cfg)
    learner = Learner(cfg)
    learner.run()

if __name__ == "__main__":
    main()