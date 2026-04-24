#!/bin/bash

# ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Check if venv directory exists and remove it to prevent path conflicts
if [ -d "venv" ]; then
    echo -e "${BLUE}▶ Existing venv found. Removing old virtual environment...${NC}"
    rm -rf venv
fi

echo -e "${BLUE}▶ Starting virtual environment (venv) creation...${NC}"

# Create new virtual environment
python3 -m venv venv

# Check if venv creation was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}✘ Failed to create virtual environment.${NC}"
    exit 1
fi

source venv/bin/activate

echo -e "${BLUE}▶ Updating pip and installing libraries...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}✅ Installation completed successfully!${NC}"