#!/bin/bash
# Setup script for vLLM environment
# This script creates a dedicated virtual environment for vLLM to avoid dependency conflicts

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up vLLM environment for HADES...${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

# Create a virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv .venv

# Activate the virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${YELLOW}Installing dependencies from requirements-vllm.txt...${NC}"
pip install -r requirements-vllm.txt

# Check if CUDA is available
echo -e "${YELLOW}Checking CUDA availability...${NC}"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Download ModernBERT model
echo -e "${YELLOW}Downloading ModernBERT model (this may take a while)...${NC}"
python -c "from huggingface_hub import snapshot_download; snapshot_download('answerdotai/ModernBERT-base', local_dir='./models/modernbert')"

# Create an activation script
echo -e "${YELLOW}Creating activation script...${NC}"
cat > activate_vllm_env.sh << 'EOF'
#!/bin/bash
# Activate the vLLM environment
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "vLLM environment activated. Run 'deactivate' to exit."
EOF

chmod +x activate_vllm_env.sh

echo -e "${GREEN}vLLM environment setup complete!${NC}"
echo -e "${GREEN}To activate the environment, run:${NC}"
echo -e "${YELLOW}source activate_vllm_env.sh${NC}"
echo -e "${GREEN}To run HADES with vLLM, use:${NC}"
echo -e "${YELLOW}python hades_cli.py --embedding-model vllm [command] [options]${NC}"

# Deactivate the virtual environment
deactivate
