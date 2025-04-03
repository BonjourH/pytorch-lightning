from xinnovation.src.core import LightningProject
from xinnovation.src.core.config import Config
import lightning as L
import argparse
from xinnovation.examples.detector4D import Sparse4DModule
from xinnovation.src.components.trainer import LightningTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a 4D object detector')
    parser.add_argument('--config_file', type=str, help='Path to the configuration file')
    parser.add_argument('--work_dir', type=str, help='Path to the working directory')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--gpu_ids', type=str, help='GPU IDs to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mode', type=str, default='train', help='[train, test, predict]')
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config.from_file(args.config_file).to_dict()

    L.seed_everything(args.seed, workers=True)

    project = LightningProject(config)

    if args.mode == 'train':
        project.train()
    elif args.mode == 'test':
        project.test()
    elif args.mode == 'predict':
        project.predict()

if __name__ == '__main__':
    main()

