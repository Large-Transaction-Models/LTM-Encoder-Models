import logging
from os.path import join, basename
from os import makedirs
import re
import os

def get_data_path(args):
    # Set the data path based on the specified dataset:
    data_path = "/data/IDEA_DeFi_Research/Data/"
    
    if args.dataset == 'Aave_V2_Mainnet':
        data_path += 'Lending_Protocols/Aave/V2/Mainnet/'
    elif args.dataset == 'Aave_V2_Polygon':
        data_path += 'Lending_Protocols/Aave/V2/Polygon/'
    elif args.dataset == 'Aave_V2_Avalanche':
        data_path += 'Lending_Protocols/Aave/V2/Avalanche/'
    elif args.dataset == 'Aave_V3_Arbitrum':
        data_path += 'Lending_Protocols/Aave/V3/Arbitrum/'
    elif args.dataset == 'Aave_V3_Avalanche':
        data_path += 'Lending_Protocols/Aave/V3/Avalanche/'
    elif args.dataset == 'Aave_V3_Fantom':
        data_path += 'Lending_Protocols/Aave/V3/Fantom/'
    elif args.dataset == 'Aave_V3_Harmony':
        data_path += 'Lending_Protocols/Aave/V3/Harmony/'
    elif args.dataset == 'Aave_V3_Optimism':
        data_path += 'Lending_Protocols/Aave/V3/Optimism/'
    elif args.dataset == 'Aave_V3_Polygon':
        data_path += 'Lending_Protocols/Aave/V3/Polygon/'
    elif args.dataset == 'Aave_V3_Mainnet':
        data_path += 'Lending_Protocols/Aave/V3/Mainnet/'
        
    elif args.dataset == 'AML_LI_Small':
        data_path += 'AML/LI_Small/'
    elif args.dataset == 'AML_LI_Medium':
        data_path += 'AML/LI_Medium/'
    elif args.dataset == 'AML_LI_Large':
        data_path += 'AML/LI_Large/'
    elif args.dataset == 'AML_HI_Small':
        data_path += 'AML/HI_Small/'
    elif args.dataset == 'AML_HI_Medium':
        data_path += 'AML/HI_Medium/'
    elif args.dataset == 'AML_HI_Large':
        data_path += 'AML/HI_Large/'
        
    elif args.dataset == 'electronics':
        data_path += 'eCommerce/Electronics/'
    elif args.dataset == 'cosmetics':
        data_path += 'eCommerce/Cosmetics/'
        
    elif args.dataset == 'Uni_V2':
        data_path += 'Decentralized_Exchanges/Uniswap/V2/'
    elif args.dataset == 'Uni_V3':
        data_path += 'Decentralized_Exchanges/Uniswap/V3/'
    
    feature_extension = ""
    if args.include_user_features==True:
        feature_extension += "_user"
    if args.include_market_features==True:
        feature_extension += "_market"
    if args.include_time_features==True:
        feature_extension += "_time"
    if args.include_exo_features==True:
        feature_extension += "_exoLagged"

    return data_path, feature_extension


def setup_logging(log_dir="logs", log_file_name='output.log'):
    makedirs(log_dir, exist_ok=True)
    log_file = join(log_dir, log_file_name)

    logger = logging.getLogger()

    if logger.hasHandlers():
        logger.handlers.clear()

    fhandler = logging.FileHandler(log_file)
    fhandler.setLevel(logging.DEBUG)

    chandler = logging.StreamHandler()
    chandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    chandler.setFormatter(formatter)

    logger.addHandler(fhandler)
    logger.addHandler(chandler)
    logger.setLevel(logging.DEBUG)

    return logger

def find_latest_checkpoint(folder_path: str) -> str:
    """
    Finds the latest checkpoint directory within the given folder path.

    Args:
        folder_path (str): The path to the directory containing checkpoints.

    Returns:
        str: The path to the latest checkpoint directory.

    Raises:
        FileNotFoundError: If no checkpoints are found.
    """
    # Check if 'final_model' directory exists
    final_model_path = os.path.join(folder_path, "final_model")
    if os.path.exists(final_model_path) and os.path.isdir(final_model_path):
        return final_model_path

    # Regex to match checkpoint directories
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)")

    checkpoints = []
    for dirname in os.listdir(folder_path):
        match = re.match(r"checkpoint-(\d+)", dirname)
        if match and os.path.isdir(os.path.join(folder_path, dirname)):
            checkpoint_number = int(match.group(1))
            checkpoints.append((checkpoint_number, dirname))

    # No checkpoint directories found
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in the given folder.")

    # Find the checkpoint with the highest number
    latest_checkpoint_dir = max(checkpoints, key=lambda x: x[0])[1]

    return os.path.join(folder_path, latest_checkpoint_dir)