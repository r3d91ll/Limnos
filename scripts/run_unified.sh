#!/bin/bash
# Unified launcher for PathRAG Dataset Builder with optional RAM disk support
# Usage: ./scripts/run_unified.sh [--gpu] [--pathrag|--graphrag|--literag] [--threads N] [--clean_db]
# Note: CPU mode is the default (no need to specify --cpu)

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_PATH=$(which python3)

# Default options
PROCESSING_MODE="cpu"  # CPU is the default processing mode
RAG_IMPL="pathrag"
CONFIG_DIR="config.d"
THREADS=24
CLEAN_DB=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      PROCESSING_MODE="gpu"
      shift
      ;;
    --pathrag)
      RAG_IMPL="pathrag"
      shift
      ;;
    --graphrag)
      RAG_IMPL="graphrag"
      shift
      ;;
    --literag)
      RAG_IMPL="literag"
      shift
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --clean_db)
      CLEAN_DB=true
      shift
      ;;
    --clean) # For backward compatibility
      echo "Warning: --clean is deprecated, use --clean_db instead"
      CLEAN_DB=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--cpu|--gpu] [--pathrag|--graphrag|--literag] [--threads N] [--clean_db]"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$PROCESSING_MODE" ]; then
    echo "Error: Processing mode (--cpu or --gpu) must be specified."
    echo "Usage: $0 [--cpu|--gpu] [--pathrag|--graphrag|--literag] [--threads N] [--clean]"
    exit 1
fi

# Configuration paths
MAIN_CONFIG="${PROJECT_ROOT}/config/config.yaml"
CONFIG_D_DIR="${PROJECT_ROOT}/${CONFIG_DIR}"

# Load shell-specific configuration for RAM disk
SHELL_CONFIG="${PROJECT_ROOT}/scripts/ramdisk_config.sh"
echo "Loading RAM disk configuration from ${SHELL_CONFIG}..."

# Set default values
USE_RAMDISK=false

# Source the shell configuration file
if [ -f "${SHELL_CONFIG}" ]; then
    source "${SHELL_CONFIG}"
else
    echo "Warning: Shell configuration file not found. Using defaults."
    # Default values
    SOURCE_RAMDISK_SIZE=20G
    DB_RAMDISK_SIZE=30G
    SOURCE_RAMDISK_PATH="/tmp/ramdisk_source_documents"
    DB_RAMDISK_PATH="/tmp/ramdisk_rag_databases"
    ENABLE_RAMDISK=false
    USE_RAMDISK_CPU=false
    USE_RAMDISK_GPU=false
    SOURCE_DIR="../source_documents"
    OUTPUT_DIR="../rag_databases"
    PYTHON_PATH=$(which python3)
fi
    
# Resolve relative paths
if [[ "${OUTPUT_DIR}" == "../"* ]]; then
    DB_DIR="${PROJECT_ROOT}/../${OUTPUT_DIR#../}"
elif [[ "${OUTPUT_DIR}" == "./"* ]]; then
    DB_DIR="${PROJECT_ROOT}/${OUTPUT_DIR#./}"
elif [[ "${OUTPUT_DIR}" == /* ]]; then
    DB_DIR="${OUTPUT_DIR}"
else
    DB_DIR="${PROJECT_ROOT}/../${OUTPUT_DIR}"
fi

if [[ "${SOURCE_DIR}" == "../"* ]]; then
    SOURCE_DIR="${PROJECT_ROOT}/../${SOURCE_DIR#../}"
elif [[ "${SOURCE_DIR}" == "./"* ]]; then
    SOURCE_DIR="${PROJECT_ROOT}/${SOURCE_DIR#./}"
elif [[ "${SOURCE_DIR}" == /* ]]; then
    SOURCE_DIR="${SOURCE_DIR}"
else
    SOURCE_DIR="${PROJECT_ROOT}/../${SOURCE_DIR}"
fi

# Check if we should use a custom output directory from config.yaml or append RAG implementation
# If CUSTOM_OUTPUT_DIR is set to true, use DB_DIR as is without appending RAG_IMPL
CUSTOM_OUTPUT_DIR=true

# Export the CUSTOM_OUTPUT_DIR environment variable for the Python process
export CUSTOM_OUTPUT_DIR="$CUSTOM_OUTPUT_DIR"

if [ "$CUSTOM_OUTPUT_DIR" = "false" ]; then
    # Append RAG implementation to output directory if not already included
    if [[ ! "${DB_DIR}" == *"${RAG_IMPL}"* ]]; then
        DB_DIR="${DB_DIR}/${RAG_IMPL}"
    fi
fi

# Check if RAM disk should be used for the current processing mode
if [ "$ENABLE_RAMDISK" = "true" ]; then
    if [ "$PROCESSING_MODE" = "cpu" ] && [ "$USE_RAMDISK_CPU" = "true" ]; then
        USE_RAMDISK=true
    elif [ "$PROCESSING_MODE" = "gpu" ] && [ "$USE_RAMDISK_GPU" = "true" ]; then
        USE_RAMDISK=true
    fi
fi

echo "Using configuration:"
echo "Source RAM disk: ${SOURCE_RAMDISK_SIZE} at ${SOURCE_RAMDISK_PATH}"
echo "Output RAM disk: ${DB_RAMDISK_SIZE} at ${DB_RAMDISK_PATH}"
echo "Source directory: ${SOURCE_DIR}"
echo "Output directory: ${DB_DIR}"
echo "RAG Implementation: ${RAG_IMPL}"
echo "Processing mode: ${PROCESSING_MODE}"
echo "Threads: ${THREADS}"
echo "Global RAM disk enabled: ${ENABLE_RAMDISK}"
echo "Using RAM disk for this run: ${USE_RAMDISK}"
echo "Clean database: ${CLEAN_DB}"

# Set up timestamp and logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/../logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/embedding_${RAG_IMPL}_${PROCESSING_MODE}_${TIMESTAMP}.log"
touch "${LOG_FILE}"

# Create directories if they don't exist (always do this regardless of RAM disk setting)
mkdir -p ${SOURCE_DIR}
mkdir -p ${DB_DIR}

# Clean existing database if requested (do this before RAM disk setup)
if [ "$CLEAN_DB" = true ]; then
    echo "Cleaning existing database in persistent storage..." | tee -a ${LOG_FILE}
    rm -rf ${DB_DIR}/*
fi

if [ "$USE_RAMDISK" = "true" ]; then
    # Create source documents RAM disk
    echo "Creating ${SOURCE_RAMDISK_SIZE} RAM disk for source documents at ${SOURCE_RAMDISK_PATH}..." | tee -a ${LOG_FILE}
    sudo mkdir -p ${SOURCE_RAMDISK_PATH}
    sudo mount -t tmpfs -o size=${SOURCE_RAMDISK_SIZE} tmpfs ${SOURCE_RAMDISK_PATH}
    sudo chmod 777 ${SOURCE_RAMDISK_PATH}
    
    # Create output database RAM disk
    echo "Creating ${DB_RAMDISK_SIZE} RAM disk for output database at ${DB_RAMDISK_PATH}..." | tee -a ${LOG_FILE}
    sudo mkdir -p ${DB_RAMDISK_PATH}
    sudo mount -t tmpfs -o size=${DB_RAMDISK_SIZE} tmpfs ${DB_RAMDISK_PATH}
    sudo chmod 777 ${DB_RAMDISK_PATH}
else
    echo "RAM disk is disabled for ${PROCESSING_MODE} mode. Using disk storage directly." | tee -a ${LOG_FILE}
fi

# Set up clean exit handler
function cleanup {
    echo "Cleaning up..." | tee -a ${LOG_FILE}
    
    if [ "$USE_RAMDISK" = "true" ]; then
        # Ensure all Python processes are terminated
        echo "Ensuring all Python processes are terminated..." | tee -a ${LOG_FILE}
        # Find any Python processes related to our script and terminate them
        PYTHON_PIDS=$(ps aux | grep "[p]ython.*src.main" | awk '{print $2}')
        if [ -n "$PYTHON_PIDS" ]; then
            echo "Terminating Python processes: $PYTHON_PIDS" | tee -a ${LOG_FILE}
            for pid in $PYTHON_PIDS; do
                sudo kill -15 $pid 2>/dev/null || true
            done
            # Give processes time to terminate gracefully
            sleep 2
            # Force kill any remaining processes
            for pid in $PYTHON_PIDS; do
                sudo kill -9 $pid 2>/dev/null || true
            done
        fi
        
        # Stop lsyncd processes
        echo "Stopping lsyncd processes..." | tee -a ${LOG_FILE}
        sudo pkill -f "lsyncd.*${SOURCE_RAMDISK_PATH}" || true
        sudo pkill -f "lsyncd.*${DB_RAMDISK_PATH}" || true
        
        # Wait for lsyncd to fully terminate
        sleep 2
        
        # Final sync from RAM disks to disk with retries
        echo "Final sync from RAM disks to disk..." | tee -a ${LOG_FILE}
        mkdir -p $(dirname "${SOURCE_DIR}")
        mkdir -p $(dirname "${DB_DIR}")
        
        # Sync source documents with retries
        MAX_RETRIES=3
        RETRY_COUNT=0
        SYNC_SUCCESS=false
        
        while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ "$SYNC_SUCCESS" = "false" ]; do
            echo "Syncing source documents (attempt $((RETRY_COUNT+1))/$MAX_RETRIES)..." | tee -a ${LOG_FILE}
            # Don't use --delete flag for source documents to preserve them
            if rsync -av ${SOURCE_RAMDISK_PATH}/ ${SOURCE_DIR}/ | tee -a ${LOG_FILE}; then
                SYNC_SUCCESS=true
            else
                echo "Source sync failed, retrying in 2 seconds..." | tee -a ${LOG_FILE}
                sleep 2
                RETRY_COUNT=$((RETRY_COUNT+1))
            fi
        done
        
        # Reset for output directory sync
        SYNC_SUCCESS=false
        RETRY_COUNT=0
        
        # If we have a subdirectory, sync it specifically to preserve the structure
        if [ -n "$OUTPUT_SUBDIR" ] && [ -d "${DB_RAMDISK_PATH}/${OUTPUT_SUBDIR}" ]; then
            echo "Syncing subdirectory ${OUTPUT_SUBDIR} from RAM disk to disk..." | tee -a ${LOG_FILE}
            mkdir -p "${DB_DIR}/${OUTPUT_SUBDIR}"
            
            while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ "$SYNC_SUCCESS" = "false" ]; do
                echo "Syncing output subdirectory (attempt $((RETRY_COUNT+1))/$MAX_RETRIES)..." | tee -a ${LOG_FILE}
                if rsync -av --delete "${DB_RAMDISK_PATH}/${OUTPUT_SUBDIR}/" "${DB_DIR}/${OUTPUT_SUBDIR}/" | tee -a ${LOG_FILE}; then
                    SYNC_SUCCESS=true
                else
                    echo "Output sync failed, retrying in 2 seconds..." | tee -a ${LOG_FILE}
                    sleep 2
                    RETRY_COUNT=$((RETRY_COUNT+1))
                fi
            done
        else
            # Otherwise sync the entire RAM disk
            while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ "$SYNC_SUCCESS" = "false" ]; do
                echo "Syncing entire output directory (attempt $((RETRY_COUNT+1))/$MAX_RETRIES)..." | tee -a ${LOG_FILE}
                if rsync -av --delete ${DB_RAMDISK_PATH}/ ${DB_DIR}/ | tee -a ${LOG_FILE}; then
                    SYNC_SUCCESS=true
                else
                    echo "Output sync failed, retrying in 2 seconds..." | tee -a ${LOG_FILE}
                    sleep 2
                    RETRY_COUNT=$((RETRY_COUNT+1))
                fi
            done
        fi
        
        # Ensure all file operations are complete before unmounting
        sync
        sleep 2
        
        # Unmount RAM disks
        echo "Unmounting RAM disks..." | tee -a ${LOG_FILE}
        # Check if mount points are actually mounted before attempting to unmount
        if mountpoint -q "${SOURCE_RAMDISK_PATH}"; then
            echo "Unmounting ${SOURCE_RAMDISK_PATH}..." | tee -a ${LOG_FILE}
            # Try regular unmount first
            if ! sudo umount "${SOURCE_RAMDISK_PATH}" 2>/dev/null; then
                echo "Regular unmount failed, waiting 5 seconds and trying again..." | tee -a ${LOG_FILE}
                sleep 5
                # Try again after waiting
                if ! sudo umount "${SOURCE_RAMDISK_PATH}" 2>/dev/null; then
                    echo "Second attempt failed, using lazy unmount..." | tee -a ${LOG_FILE}
                    sudo umount -l "${SOURCE_RAMDISK_PATH}" || echo "Failed to unmount ${SOURCE_RAMDISK_PATH}" | tee -a ${LOG_FILE}
                fi
            fi
        else
            echo "${SOURCE_RAMDISK_PATH} is not mounted" | tee -a ${LOG_FILE}
        fi
        
        if mountpoint -q "${DB_RAMDISK_PATH}"; then
            echo "Unmounting ${DB_RAMDISK_PATH}..." | tee -a ${LOG_FILE}
            # Try regular unmount first
            if ! sudo umount "${DB_RAMDISK_PATH}" 2>/dev/null; then
                echo "Regular unmount failed, waiting 5 seconds and trying again..." | tee -a ${LOG_FILE}
                sleep 5
                # Try again after waiting
                if ! sudo umount "${DB_RAMDISK_PATH}" 2>/dev/null; then
                    echo "Second attempt failed, using lazy unmount..." | tee -a ${LOG_FILE}
                    sudo umount -l "${DB_RAMDISK_PATH}" || echo "Failed to unmount ${DB_RAMDISK_PATH}" | tee -a ${LOG_FILE}
                fi
            fi
        else
            echo "${DB_RAMDISK_PATH} is not mounted" | tee -a ${LOG_FILE}
        fi
        
        echo "RAM disk cleanup complete." | tee -a ${LOG_FILE}
    else
        echo "No RAM disk cleanup needed." | tee -a ${LOG_FILE}
    fi
}

# Register cleanup handler for various exit signals
trap cleanup EXIT INT TERM

# Extract the subdirectory from config.yaml output_dir if it exists
# This improved regex extracts only the path, ignoring any comments that might follow
CONFIG_OUTPUT_DIR=$(grep -E "^output_dir:" "${MAIN_CONFIG}" | sed -E 's/^output_dir:[[:space:]]*"?([^"#]+)"?[[:space:]]*#?.*/\1/' | tr -d ' ')
OUTPUT_SUBDIR=""

if [[ "$CONFIG_OUTPUT_DIR" == */* ]]; then
    # Extract the subdirectory part after the last slash
    OUTPUT_SUBDIR=$(echo "$CONFIG_OUTPUT_DIR" | sed -E 's/.*\/([^\/]+)$/\1/')
    echo "Detected output subdirectory: $OUTPUT_SUBDIR" | tee -a ${LOG_FILE}
fi

# Now that all variables are defined, we can start the administrative tasks
if [ "$USE_RAMDISK" = "true" ]; then
    # Create all necessary RAM disk directories
    echo "Creating all necessary RAM disk directories..." | tee -a ${LOG_FILE}
    mkdir -p ${SOURCE_RAMDISK_PATH}
    mkdir -p ${DB_RAMDISK_PATH}
    
    # Create subdirectories for the specific RAG implementation if OUTPUT_SUBDIR is defined
    if [ -n "$OUTPUT_SUBDIR" ]; then
        echo "Creating RAM disk subdirectories for ${OUTPUT_SUBDIR}..." | tee -a ${LOG_FILE}
        mkdir -p "${DB_RAMDISK_PATH}/${OUTPUT_SUBDIR}"
        mkdir -p "${DB_RAMDISK_PATH}/${OUTPUT_SUBDIR}/embeddings"
        mkdir -p "${DB_RAMDISK_PATH}/${OUTPUT_SUBDIR}/graph"
        mkdir -p "${DB_RAMDISK_PATH}/${OUTPUT_SUBDIR}/chunks"
        mkdir -p "${DB_RAMDISK_PATH}/${OUTPUT_SUBDIR}/logs"
        mkdir -p "${DB_RAMDISK_PATH}/${OUTPUT_SUBDIR}/metadata"
        
        # Also ensure the directories exist in the disk location
        mkdir -p "${DB_DIR}/${OUTPUT_SUBDIR}"
        mkdir -p "${DB_DIR}/${OUTPUT_SUBDIR}/embeddings"
        mkdir -p "${DB_DIR}/${OUTPUT_SUBDIR}/graph"
        mkdir -p "${DB_DIR}/${OUTPUT_SUBDIR}/chunks"
        mkdir -p "${DB_DIR}/${OUTPUT_SUBDIR}/logs"
        mkdir -p "${DB_DIR}/${OUTPUT_SUBDIR}/metadata"
    fi
    
    # First, copy source documents to RAM disk (faster initial copy than lsyncd)
    echo "Copying source documents to RAM disk..." | tee -a ${LOG_FILE}
    echo "Start time: $(date)" | tee -a ${LOG_FILE}
    mkdir -p ${SOURCE_DIR}
    rsync -av --progress ${SOURCE_DIR}/ ${SOURCE_RAMDISK_PATH}/ 2>&1 | tee -a ${LOG_FILE}
    echo "Copy completed at: $(date)" | tee -a ${LOG_FILE}
    
    # Create output directory structure in RAM disk
    echo "Creating output directory structure in RAM disk..." | tee -a ${LOG_FILE}
    
    # Clean RAM disk if we're cleaning the database
    if [ "$CLEAN_DB" = true ]; then
        echo "Cleaning existing database in RAM disk..." | tee -a ${LOG_FILE}
        rm -rf ${DB_RAMDISK_PATH}/*
    fi
    
    # Set up lsyncd for bidirectional sync
    echo "Setting up lsyncd for bidirectional sync..." | tee -a ${LOG_FILE}
    
    # Create lsyncd config for source documents
    SOURCE_LSYNCD_CONFIG="/tmp/lsyncd_source_${TIMESTAMP}.conf"
    cat > ${SOURCE_LSYNCD_CONFIG} << EOF
settings {
    logfile = "/tmp/lsyncd_source_${TIMESTAMP}.log",
    statusFile = "/tmp/lsyncd_source_${TIMESTAMP}.status",
    nodaemon = false
}

sync {
    default.rsync,
    source = "${SOURCE_RAMDISK_PATH}/",
    target = "${SOURCE_DIR}/",
    delay = 5,
    rsync = {
        archive = true,
        verbose = true
    }
}

sync {
    default.rsync,
    source = "${SOURCE_DIR}/",
    target = "${SOURCE_RAMDISK_PATH}/",
    delay = 5,
    rsync = {
        archive = true,
        verbose = true
    }
}
EOF
    
    # Create lsyncd config for output database
    DB_LSYNCD_CONFIG="/tmp/lsyncd_db_${TIMESTAMP}.conf"
    cat > ${DB_LSYNCD_CONFIG} << EOF
settings {
    logfile = "/tmp/lsyncd_db_${TIMESTAMP}.log",
    statusFile = "/tmp/lsyncd_db_${TIMESTAMP}.status",
    nodaemon = false
}

sync {
    default.rsync,
    source = "${DB_RAMDISK_PATH}/",
    target = "${DB_DIR}/",
    delay = 5,
    rsync = {
        archive = true,
        verbose = true
    }
}

sync {
    default.rsync,
    source = "${DB_DIR}/",
    target = "${DB_RAMDISK_PATH}/",
    delay = 5,
    rsync = {
        archive = true,
        verbose = true
    }
}
EOF

    # Start lsyncd processes
    echo "Starting lsyncd processes..." | tee -a ${LOG_FILE}
    sudo lsyncd ${SOURCE_LSYNCD_CONFIG}
    sudo lsyncd ${DB_LSYNCD_CONFIG}
    echo "Lsyncd processes started." | tee -a ${LOG_FILE}
else
    echo "Skipping lsyncd setup (RAM disk disabled)" | tee -a ${LOG_FILE}
    
    # Clean existing database if requested
    if [ "$CLEAN_DB" = true ]; then
        echo "Cleaning existing database on disk..." | tee -a ${LOG_FILE}
        rm -rf ${DB_DIR}/*
        mkdir -p ${DB_DIR}
    fi
fi

# Set environment variables for PyTorch
echo "Setting environment variables for PyTorch..." | tee -a ${LOG_FILE}
export OMP_NUM_THREADS=${THREADS}
export MKL_NUM_THREADS=${THREADS}
export NUMEXPR_NUM_THREADS=${THREADS}
export NUMEXPR_MAX_THREADS=${THREADS}

# Ensure Python environment is preserved when running with sudo
if [ -d "/home/todd/ML-Lab/New-HADES/.venv" ]; then
    export PYTHONPATH="/home/todd/ML-Lab/New-HADES:$PYTHONPATH"
    # If running as sudo, make sure to use the virtual environment
    if [ "$EUID" -eq 0 ]; then
        echo "Running as root, ensuring virtual environment is used..." | tee -a ${LOG_FILE}
        source "/home/todd/ML-Lab/New-HADES/.venv/bin/activate"
    fi
fi

# Set up config dir argument
CONFIG_DIR_ARG="--config_dir ${CONFIG_D_DIR}"

# Set up processing mode argument and environment variables for Ollama
if [ "$PROCESSING_MODE" = "cpu" ]; then
    # CPU is now the default, so no argument needed
    PROCESSING_ARG=""
    
    # Set environment variables to force Ollama to use CPU
    export OLLAMA_USE_CPU=true
    export CUDA_VISIBLE_DEVICES=""  # Hide all CUDA devices
    echo "Setting environment variables to force CPU usage for Ollama" | tee -a ${LOG_FILE}
else
    PROCESSING_ARG="--gpu"
    
    # Set environment variables to allow Ollama to use GPU
    export OLLAMA_USE_CPU=false
    unset CUDA_VISIBLE_DEVICES  # Allow all CUDA devices
    echo "Setting environment variables to enable GPU usage for Ollama" | tee -a ${LOG_FILE}
fi

# Set up RAG implementation argument
RAG_IMPL_ARG="--${RAG_IMPL}"

# We'll use a temporary config directory that includes all regular config files plus our path overrides
TEMP_CONFIG_DIR="${PROJECT_ROOT}/config.d/temp"

# Remove any existing temp directory and create a fresh one
rm -rf "${TEMP_CONFIG_DIR}"
mkdir -p "${TEMP_CONFIG_DIR}"

# Copy all existing config files from config.d to our temp directory
cp "${CONFIG_D_DIR}"/*.yaml "${TEMP_CONFIG_DIR}/"

# Create a temporary config file for source and output directories
# Using 99- prefix ensures it loads last and overrides any previous settings
TEMP_CONFIG_FILE="${TEMP_CONFIG_DIR}/99-paths.yaml"

# This section has been moved to the top of the script for better variable definition

# Determine which paths to use based on RAM disk setting
if [ "$USE_RAMDISK" = "true" ]; then
    echo "# Temporary path configuration for RAM disk" > "${TEMP_CONFIG_FILE}"
    echo "source_documents: \"${SOURCE_RAMDISK_PATH}\"" >> "${TEMP_CONFIG_FILE}"
    
    # Use the subdirectory in RAM disk if specified
    if [ -n "$OUTPUT_SUBDIR" ]; then
        echo "output_dir: \"${DB_RAMDISK_PATH}/${OUTPUT_SUBDIR}\"" >> "${TEMP_CONFIG_FILE}"
    else
        echo "output_dir: \"${DB_RAMDISK_PATH}\"" >> "${TEMP_CONFIG_FILE}"
    fi
    
    echo "Using RAM disk paths for processing" | tee -a ${LOG_FILE}
else
    echo "# Temporary path configuration for disk storage" > "${TEMP_CONFIG_FILE}"
    echo "source_documents: \"${SOURCE_DIR}\"" >> "${TEMP_CONFIG_FILE}"
    echo "output_dir: \"${DB_DIR}\"" >> "${TEMP_CONFIG_FILE}"
    echo "Using disk storage paths for processing" | tee -a ${LOG_FILE}
fi

# Run the dataset builder
echo "Running PathRAG Dataset Builder with options: ${PROCESSING_ARG} ${RAG_IMPL_ARG} --threads ${THREADS}" | tee -a ${LOG_FILE}
echo "Using config directory: ${TEMP_CONFIG_DIR}" | tee -a ${LOG_FILE}
echo "Command: ${PYTHON_PATH} -m src.main --config_dir ${TEMP_CONFIG_DIR} ${PROCESSING_ARG} ${RAG_IMPL_ARG} --threads ${THREADS}" | tee -a ${LOG_FILE}

${PYTHON_PATH} -m src.main --config_dir ${TEMP_CONFIG_DIR} ${PROCESSING_ARG} ${RAG_IMPL_ARG} --threads ${THREADS}

# Clean up temporary config
rm -f "${TEMP_CONFIG_FILE}"

echo "PathRAG Dataset Builder completed." | tee -a ${LOG_FILE}
