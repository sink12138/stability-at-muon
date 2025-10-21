#!/usr/bin/env python

import argparse
import sys
import os
import subprocess

def run_training(args):
    """Run training module"""
    script_path = os.path.join('src', 'training', 'train.py')
    
    cmd = ['python3', script_path] + args.training_args
    subprocess.run(cmd)

def run_testing(args):
    """Run testing module"""
    if args.test_type == 'blackbox':
        script_path = os.path.join('src', 'testing', 'test_blackbox.py')
    else:
        script_path = os.path.join('src', 'testing', 'test_transfer.py')
    
    cmd = ['python3', script_path] + args.testing_args
    subprocess.run(cmd)

def run_visualization(args):
    """Run visualization module"""
    if args.viz_type == 'compare':
        script_path = os.path.join('src', 'visualization', 'compare.py')
    else:
        script_path = os.path.join('src', 'visualization', 'visualize.py')
    
    cmd = ['python3', script_path] + args.viz_args
    subprocess.run(cmd)

def run_experiments(args):
    """Run experiments"""
    script_path = os.path.join('src', 'utils', 'run_experiments.py')
    cmd = ['python3', script_path]
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description='Stability at Muon')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Run training')
    train_parser.add_argument('--method', choices=['vanilla', 'fast', 'free'], 
                             default='vanilla', help='Training method')
    train_parser.add_argument('training_args', nargs='*', 
                             help='Pass additional arguments to the training script')
    train_parser.set_defaults(func=run_training)
    
    # Testing command
    test_parser = subparsers.add_parser('test', help='Run testing')
    test_parser.add_argument('--test_type', choices=['blackbox', 'transfer'], 
                            default='blackbox', help='Testing type')
    test_parser.add_argument('testing_args', nargs='*', 
                            help='Pass additional arguments to the testing script')
    test_parser.set_defaults(func=run_testing)
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Run visualization')
    viz_parser.add_argument('--viz_type', choices=['compare', 'visualize'], 
                           default='compare', help='Visualization type')
    viz_parser.add_argument('viz_args', nargs='*', 
                           help='Pass additional arguments to the visualization script')
    viz_parser.set_defaults(func=run_visualization)
    
    # Experiments command
    exp_parser = subparsers.add_parser('experiments', help='Run experiments')
    exp_parser.set_defaults(func=run_experiments)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()