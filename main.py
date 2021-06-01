#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse

from SampleTraining import SampleTraining

# Configure the parser
parser = argparse.ArgumentParser()

parser.add_argument(
    '--debug',
    action='store_true',
    dest='debug',
    help='Run in debug mode'
)

# Parse the arguments
args = parser.parse_args()

# Create the service
server = SampleTraining(args.debug)
server.trainModel()
