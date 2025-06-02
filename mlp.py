from genericpath import exists
from Training import Training
from Predicting import Predicting 
import argparse
import json
import os.path

def main():
    """
    --mode train/predict mandatory
        Train:
            - arg is a json filename:
                - checks if file exists.
                - checks args in file, if one isnt there, uses default values.
            - arg is --epochs --learning_rate --topology:
                - if an arg is not given, uses the default values
    """
    parser = argparse.ArgumentParser(prog='Multilayer Perceptron', description='Create and train a new MLP model or do predictions with an existing one.')
    parser.add_argument("--mode", "-m", choices=["train", "predict"], type=str, required=True)
    parser.add_argument("--config", "-c", type=str, required=False)
    parser.add_argument("--filename", "-f", type=str, required=False)
    parser.add_argument("--learning_rate", "-lr", type=float, required=False)
    parser.add_argument("--epochs", "-e", type=int, required=False)
    parser.add_argument("--topology", "-t", type=int, nargs="+", required=False)
    args = parser.parse_args()
    if args.mode == "train" and not args.config and not arg_checker(args):
        parser.print_help()
        return -1
    if args.mode == "train":
        if args.config:
            try:
                epochs, topology, lr, filename = get_infos_from_json(args.config)
            except:
                print(f"ERROR : configuration file \'{args.config}\' contains an error.")
                return (-1)
            print(epochs, type(epochs), topology, type(topology), lr, type(lr), filename, type(filename))
            t = Training(epochs=epochs, neural_network_list=topology, learning_rate=lr, model_filename=filename)
        else:
            t = Training(epochs=args.epochs, neural_network_list=args.topology, learning_rate=args.learning_rate, model_filename=args.filename)
        t.train()
    elif args.mode == "predict":
        try:
            p = Predicting(model_filename=args.filename)
            p.predict()
        except:
            print("ERROR : prediction model invalid.")
            exit(-1)
    else:
        parser.print_help()

def get_infos_from_json(filename):
    """
    Uses a json file to get the arguments of a training session.
    Checks if the config file contains all the needed informations.
    """
    if os.path.exists(filename) is False:
        print(f"ERROR : configuration file \'{filename}\' does not exist.")
        exit(-1)
    try:
        with open(filename) as json_file:
            dict = json.load(json_file)
    except:
        print(f"ERROR : configuration file \'{filename}\' is broken.")
        exit(-1)
    keys_to_test = ["epochs", "topology", "learning_rate", "filename"]
    if not all_values_are_keys(keys_to_test, dict):
        print("Error: model config file needs: \"epochs\", \"topology\", \"learning_rate\" and \"filename\".")
        exit(-1)
    return int(dict["epochs"]), list(dict["topology"]), float(dict["learning_rate"]), str(dict["filename"])

def all_values_are_keys(value_list, dictionary):
    """
    Checks if a dictionnary contains the needed keys.
    """
    return all(value in dictionary for value in value_list)

def arg_checker(args):
    """
    Checks if needed arguments are initialized.
    """
    l = [args.filename, args.learning_rate, args.topology, args.epochs]
    all_set = True
    for arg in l:
        if arg is None:
            all_set = False
    if args.config is None and not all_set:
        return False
    return True

if __name__ == "__main__":
    main()