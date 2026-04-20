#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}▶ Starting virtual environment (venv) creation...${NC}"

python3 -m venv venv

source venv/bin/activate

echo -e "${BLUE}▶ Updating pip and installing libraries...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}✅ Installation completed successfully!${NC}"