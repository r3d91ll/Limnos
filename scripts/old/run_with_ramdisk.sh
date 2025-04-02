#!/bin/bash
# Enhanced RAM Disk Launcher for PathRAG Dataset Builder
# This script creates RAM disks for both source documents and output database,
# syncs them with disk storage using lsyncd, and runs the dataset builder with
# configurable CPU/GPU-based embedding for maximum performance

set -e

# Default processing mode
PROCESSING_MODE="cpu"
RAG_IMPL="pathrag"
CLEAN_DATASET=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      PROCESSING_MODE="gpu"
      shift
      ;;
    --cpu)
      PROCESSING_MODE="cpu"
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
      NUM_THREADS="$2"
      shift 2
      ;;
    --clean)
      CLEAN_DATASET=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--cpu|--gpu] [--pathrag|--graphrag|--literag] [--threads N] [--clean]"
      exit 1
      ;;
  esac
done

# Set script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Use Python from the virtual environment
PYTHON_PATH="${PROJECT_ROOT}/../.venv/bin/python3"

# Configuration paths
CONFIG_D_DIR="${PROJECT_ROOT}/config.d"
CONFIG_FILE="${PROJECT_ROOT}/config/config.yaml"
NUM_THREADS=${NUM_THREADS:-24}  # Default to 24 threads if not specified

# Load RAM disk settings from main config file
MAIN_CONFIG="${PROJECT_ROOT}/config/config.yaml"

# Default values in case config file doesn't exist
SOURCE_RAMDISK_SIZE=20G
DB_RAMDISK_SIZE=30G
SOURCE_RAMDISK_PATH="/tmp/ramdisk_source_documents"
DB_RAMDISK_PATH="/tmp/ramdisk_rag_databases"

# Read settings from main config.yaml
if [ -f "${MAIN_CONFIG}" ]; then
    echo "Loading RAM disk configuration from ${MAIN_CONFIG}..."
    
    # Extract disk storage locations
    OUTPUT_DIR=$(grep -E "^output_dir:" "${MAIN_CONFIG}" | sed 's/^output_dir: *"\(.*\)"/\1/' | sed "s/^output_dir: *'\(.*\)'/\1/" | sed 's/^output_dir: *//')
    SOURCE_DIR=$(grep -E "^source_documents:" "${MAIN_CONFIG}" | sed 's/^source_documents: *"\(.*\)"/\1/' | sed "s/^source_documents: *'\(.*\)'/\1/" | sed 's/^source_documents: *//')
    
    # Extract RAM disk mount points
    SOURCE_RAMDISK_PATH=$(grep -E "source_mount:" "${MAIN_CONFIG}" | sed 's/.*source_mount: *"\(.*\)"/\1/' | sed "s/.*source_mount: *'\(.*\)'/\1/" | sed 's/.*source_mount: *//')
    DB_RAMDISK_PATH=$(grep -E "output_mount:" "${MAIN_CONFIG}" | sed 's/.*output_mount: *"\(.*\)"/\1/' | sed "s/.*output_mount: *'\(.*\)'/\1/" | sed 's/.*output_mount: *//')
    
    # Extract RAM disk sizes
    SOURCE_RAMDISK_SIZE=$(grep -E "source_size:" "${MAIN_CONFIG}" | sed 's/.*source_size: *"\(.*\)"/\1/' | sed "s/.*source_size: *'\(.*\)'/\1/" | sed 's/.*source_size: *//' | sed 's/ *#.*$//')
    DB_RAMDISK_SIZE=$(grep -E "output_size:" "${MAIN_CONFIG}" | sed 's/.*output_size: *"\(.*\)"/\1/' | sed "s/.*output_size: *'\(.*\)'/\1/" | sed 's/.*output_size: *//' | sed 's/ *#.*$//')
    
    # Extract RAM disk usage flags
    USE_RAMDISK_CPU=$(grep -E "use_for_cpu:" "${MAIN_CONFIG}" | grep -E "true" > /dev/null && echo "true" || echo "false")
    USE_RAMDISK_GPU=$(grep -E "use_for_gpu:" "${MAIN_CONFIG}" | grep -E "true" > /dev/null && echo "true" || echo "false")
    
    # Handle relative paths for disk storage locations
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
    
    if [[ "${CACHE_DIR}" == "../"* ]]; then
        CACHE_DIR="${PROJECT_ROOT}/../${CACHE_DIR#../}"
    elif [[ "${CACHE_DIR}" == "./"* ]]; then
        CACHE_DIR="${PROJECT_ROOT}/${CACHE_DIR#./}"
    elif [[ "${CACHE_DIR}" == /* ]]; then
        CACHE_DIR="${CACHE_DIR}"
    else
        CACHE_DIR="${PROJECT_ROOT}/../${CACHE_DIR}"
    fi
else
    # Default paths if config.yaml doesn't exist
    SOURCE_DIR="${PROJECT_ROOT}/../source_documents"
    DB_DIR="${PROJECT_ROOT}/../rag_databases/${RAG_IMPL}"
    CACHE_DIR="${PROJECT_ROOT}/../cache"
fi

# Parse YAML config file if it exists
if [ -f "${RAMDISK_CONFIG}" ]; then
    echo "Loading RAM disk configuration from ${RAMDISK_CONFIG}..."
    
    # Extract values using grep and sed
    if grep -q "source_size:" "${RAMDISK_CONFIG}"; then
        # Extract the value without any comments
        SOURCE_RAMDISK_SIZE=$(grep "source_size:" "${RAMDISK_CONFIG}" | sed 's/^.*source_size: *"\(.*\)"/\1/' | sed "s/^.*source_size: *'\(.*\)'/\1/" | sed 's/^.*source_size: *//' | sed 's/ *#.*$//')
    fi
    
    if grep -q "output_size:" "${RAMDISK_CONFIG}"; then
        # Extract the value without any comments
        DB_RAMDISK_SIZE=$(grep "output_size:" "${RAMDISK_CONFIG}" | sed 's/^.*output_size: *"\(.*\)"/\1/' | sed "s/^.*output_size: *'\(.*\)'/\1/" | sed 's/^.*output_size: *//' | sed 's/ *#.*$//')
    fi
    
    if grep -q "source_path:" "${RAMDISK_CONFIG}"; then
        SOURCE_RAMDISK_PATH=$(grep "source_path:" "${RAMDISK_CONFIG}" | sed 's/^.*source_path: *"\(.*\)"/\1/' | sed "s/^.*source_path: *'\(.*\)'/\1/" | sed 's/^.*source_path: *//')
    fi
    
    if grep -q "output_path:" "${RAMDISK_CONFIG}"; then
        DB_RAMDISK_PATH=$(grep "output_path:" "${RAMDISK_CONFIG}" | sed 's/^.*output_path: *"\(.*\)"/\1/' | sed "s/^.*output_path: *'\(.*\)'/\1/" | sed 's/^.*output_path: *//')
    fi
    
    # Override source and output disk paths if specified in config
    if grep -q "source_disk_path:" "${RAMDISK_CONFIG}"; then
        SOURCE_DISK_PATH=$(grep "source_disk_path:" "${RAMDISK_CONFIG}" | sed 's/^.*source_disk_path: *"\(.*\)"/\1/' | sed "s/^.*source_disk_path: *'\(.*\)'/\1/" | sed 's/^.*source_disk_path: *//')
        
        # Handle relative paths
        if [[ "${SOURCE_DISK_PATH}" == "../"* ]]; then
            SOURCE_DIR="${PROJECT_ROOT}/../${SOURCE_DISK_PATH#../}"
        elif [[ "${SOURCE_DISK_PATH}" == "./"* ]]; then
            SOURCE_DIR="${PROJECT_ROOT}/${SOURCE_DISK_PATH#./}"
        elif [[ "${SOURCE_DISK_PATH}" == /* ]]; then
            SOURCE_DIR="${SOURCE_DISK_PATH}"
        else
            SOURCE_DIR="${PROJECT_ROOT}/../${SOURCE_DISK_PATH}"
        fi
    fi
    
    if grep -q "output_disk_path:" "${RAMDISK_CONFIG}"; then
        OUTPUT_DISK_PATH=$(grep "output_disk_path:" "${RAMDISK_CONFIG}" | sed 's/^.*output_disk_path: *"\(.*\)"/\1/' | sed "s/^.*output_disk_path: *'\(.*\)'/\1/" | sed 's/^.*output_disk_path: *//')
        
        # Handle relative paths
        if [[ "${OUTPUT_DISK_PATH}" == "../"* ]]; then
            DB_DIR="${PROJECT_ROOT}/../${OUTPUT_DISK_PATH#../}"
        elif [[ "${OUTPUT_DISK_PATH}" == "./"* ]]; then
            DB_DIR="${PROJECT_ROOT}/${OUTPUT_DISK_PATH#./}"
        elif [[ "${OUTPUT_DISK_PATH}" == /* ]]; then
            DB_DIR="${OUTPUT_DISK_PATH}"
        else
            DB_DIR="${PROJECT_ROOT}/../${OUTPUT_DISK_PATH}"
        fi
    fi
fi

# Append RAG implementation to output directory if not already included
if [[ ! "${DB_DIR}" == *"${RAG_IMPL}"* ]]; then
    DB_DIR="${DB_DIR}/${RAG_IMPL}"
fi

# Check if RAM disk should be used for the current processing mode
USE_RAMDISK=false
if [ "$PROCESSING_MODE" = "cpu" ] && [ "$USE_RAMDISK_CPU" = "true" ]; then
    USE_RAMDISK=true
elif [ "$PROCESSING_MODE" = "gpu" ] && [ "$USE_RAMDISK_GPU" = "true" ]; then
    USE_RAMDISK=true
fi

echo "Using RAM disk settings:"
echo "Source RAM disk: ${SOURCE_RAMDISK_SIZE} at ${SOURCE_RAMDISK_PATH}"
echo "Output RAM disk: ${DB_RAMDISK_SIZE} at ${DB_RAMDISK_PATH}"
echo "Cache RAM disk: ${CACHE_RAMDISK_SIZE} at ${CACHE_RAMDISK_PATH}"
echo "Source directory: ${SOURCE_DIR}"
echo "Output directory: ${DB_DIR}"
echo "Cache directory: ${CACHE_DIR}"
echo "RAG Implementation: ${RAG_IMPL}"
echo "Processing mode: ${PROCESSING_MODE}"
echo "Threads: ${NUM_THREADS}"
echo "Using RAM disk: ${USE_RAMDISK}"

# Set up timestamp and logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/../logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/ramdisk_embedding_${RAG_IMPL}_${PROCESSING_MODE}_${TIMESTAMP}.log"
touch "${LOG_FILE}"

if [ "$USE_RAMDISK" = "true" ]; then
    # Create source documents RAM disk
    # Extract just the size value without any comments
    SOURCE_SIZE_CLEAN=$(echo "${SOURCE_RAMDISK_SIZE}" | sed 's/ *#.*$//')
    echo "Creating ${SOURCE_SIZE_CLEAN} RAM disk for source documents at ${SOURCE_RAMDISK_PATH}..." | tee -a ${LOG_FILE}
    sudo mkdir -p ${SOURCE_RAMDISK_PATH}
    sudo mount -t tmpfs -o size=${SOURCE_SIZE_CLEAN} tmpfs ${SOURCE_RAMDISK_PATH}
    sudo chmod 777 ${SOURCE_RAMDISK_PATH}
    
    # Create output database RAM disk
    # Extract just the size value without any comments
    DB_SIZE_CLEAN=$(echo "${DB_RAMDISK_SIZE}" | sed 's/ *#.*$//')
    echo "Creating ${DB_SIZE_CLEAN} RAM disk for output database at ${DB_RAMDISK_PATH}..." | tee -a ${LOG_FILE}
    sudo mkdir -p ${DB_RAMDISK_PATH}
    sudo mount -t tmpfs -o size=${DB_SIZE_CLEAN} tmpfs ${DB_RAMDISK_PATH}
    sudo chmod 777 ${DB_RAMDISK_PATH}
    

else
    echo "RAM disk is disabled for ${PROCESSING_MODE} mode. Using disk storage directly." | tee -a ${LOG_FILE}
    # Create directories if they don't exist
    mkdir -p ${SOURCE_DIR}
    mkdir -p ${DB_DIR}
fi

# Set up clean exit handler
function cleanup {
    echo "Cleaning up..." | tee -a ${LOG_FILE}
    
    if [ "$USE_RAMDISK" = "true" ]; then
        # Stop lsyncd processes
        echo "Stopping lsyncd processes..." | tee -a ${LOG_FILE}
        sudo pkill -f "lsyncd.*${SOURCE_RAMDISK_PATH}" || true
        sudo pkill -f "lsyncd.*${DB_RAMDISK_PATH}" || true
        
        # Final sync from RAM disks to disk
        echo "Final sync from RAM disks to disk..." | tee -a ${LOG_FILE}
        mkdir -p $(dirname "${SOURCE_DIR}")
        mkdir -p $(dirname "${DB_DIR}")
        rsync -av --delete ${SOURCE_RAMDISK_PATH}/ ${SOURCE_DIR}/ | tee -a ${LOG_FILE}
        rsync -av --delete ${DB_RAMDISK_PATH}/ ${DB_DIR}/ | tee -a ${LOG_FILE}
        
        # Unmount RAM disks
        echo "Unmounting RAM disks..." | tee -a ${LOG_FILE}
        sudo umount ${SOURCE_RAMDISK_PATH} || true
        sudo umount ${DB_RAMDISK_PATH} || true
        
        echo "RAM disk cleanup complete." | tee -a ${LOG_FILE}
    else
        echo "No RAM disk cleanup needed." | tee -a ${LOG_FILE}
    fi
}

# Register cleanup handler for various exit signals
trap cleanup EXIT INT TERM

if [ "$USE_RAMDISK" = "true" ]; then
    # First, copy source documents to RAM disk (faster initial copy than lsyncd)
    echo "Copying source documents to RAM disk..." | tee -a ${LOG_FILE}
    echo "Start time: $(date)" | tee -a ${LOG_FILE}
    mkdir -p ${SOURCE_DIR}
    rsync -av --progress ${SOURCE_DIR}/ ${SOURCE_RAMDISK_PATH}/ 2>&1 | tee -a ${LOG_FILE}
    echo "Copy completed at: $(date)" | tee -a ${LOG_FILE}
    

    
    # Create output directory structure in RAM disk
    echo "Creating output directory structure in RAM disk..." | tee -a ${LOG_FILE}
    mkdir -p ${DB_RAMDISK_PATH}
    
    # Clean existing dataset if requested
    if [ "$CLEAN_DATASET" = true ]; then
        echo "Cleaning existing dataset in ${DB_RAMDISK_PATH}/${RAG_IMPL}..." | tee -a ${LOG_FILE}
        rm -rf ${DB_RAMDISK_PATH}/${RAG_IMPL}
        mkdir -p ${DB_RAMDISK_PATH}/${RAG_IMPL}
        echo "Existing dataset cleaned." | tee -a ${LOG_FILE}
    fi
    
    # Create essential subdirectories for PathRAG in RAM disk
    echo "Creating essential subdirectories for ${RAG_IMPL} in RAM disk..." | tee -a ${LOG_FILE}
    mkdir -p ${DB_RAMDISK_PATH}/${RAG_IMPL}/chunks
    mkdir -p ${DB_RAMDISK_PATH}/${RAG_IMPL}/embeddings
    mkdir -p ${DB_RAMDISK_PATH}/${RAG_IMPL}/graphs
    mkdir -p ${DB_RAMDISK_PATH}/${RAG_IMPL}/metadata
    mkdir -p ${DB_RAMDISK_PATH}/${RAG_IMPL}/raw
    echo "Essential subdirectories created in RAM disk." | tee -a ${LOG_FILE}
else
    # When not using RAM disk, just ensure the disk directories exist
    echo "Setting up disk directories (RAM disk disabled)..." | tee -a ${LOG_FILE}
    mkdir -p ${SOURCE_DIR}
    
    # Clean existing dataset if requested
    if [ "$CLEAN_DATASET" = true ]; then
        echo "Cleaning existing dataset in ${DB_DIR}..." | tee -a ${LOG_FILE}
        rm -rf ${DB_DIR}
        mkdir -p ${DB_DIR}
        echo "Existing dataset cleaned." | tee -a ${LOG_FILE}
    fi
    
    # Create essential subdirectories for PathRAG on disk
    echo "Creating essential subdirectories for ${RAG_IMPL} on disk..." | tee -a ${LOG_FILE}
    mkdir -p ${DB_DIR}/chunks
    mkdir -p ${DB_DIR}/embeddings
    mkdir -p ${DB_DIR}/graphs
    mkdir -p ${DB_DIR}/metadata
    mkdir -p ${DB_DIR}/raw
    echo "Essential subdirectories created on disk." | tee -a ${LOG_FILE}
fi

# Always ensure the disk output directory exists
mkdir -p ${DB_DIR}

if [ "$USE_RAMDISK" = "true" ]; then
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
fi

# Set environment variables for PyTorch
echo "Setting environment variables for PyTorch..." | tee -a ${LOG_FILE}
export OMP_NUM_THREADS=${NUM_THREADS}
export MKL_NUM_THREADS=${NUM_THREADS}
export NUMEXPR_NUM_THREADS=${NUM_THREADS}
export NUMEXPR_MAX_THREADS=${NUM_THREADS}

# Set up config dir argument
CONFIG_DIR_ARG="--config_dir ${CONFIG_D_DIR}"

# Set up processing mode argument
if [ "$PROCESSING_MODE" = "cpu" ]; then
    PROCESSING_ARG="--cpu"
    CONFIG_SUFFIX="35-cpu-embedder.yaml"
    echo "Using CPU-based embedding with ${NUM_THREADS} threads" | tee -a ${LOG_FILE}
else
    PROCESSING_ARG="--gpu"
    CONFIG_SUFFIX="30-embedders.yaml"  # Default GPU config
    echo "Using GPU-based embedding" | tee -a ${LOG_FILE}
fi

# Set up RAG implementation argument
RAG_IMPL_ARG="--${RAG_IMPL}"

# Run the dataset builder with RAM disk paths
echo "Running dataset builder with RAM disk paths..." | tee -a ${LOG_FILE}
echo "Command: ${PYTHON_PATH} -m src.main ${CONFIG_DIR_ARG} ${PROCESSING_ARG} ${RAG_IMPL_ARG} --threads ${NUM_THREADS}" | tee -a ${LOG_FILE}
echo "Start time: $(date)" | tee -a ${LOG_FILE}

# Change to project directory
cd ${PROJECT_ROOT}

# Run the dataset builder
${PYTHON_PATH} -m src.main ${CONFIG_DIR_ARG} ${PROCESSING_ARG} ${RAG_IMPL_ARG} --threads ${NUM_THREADS} 2>&1 | tee -a ${LOG_FILE}

# Check if the command succeeded
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Dataset builder failed. Check the log for details." | tee -a ${LOG_FILE}
    exit 1
fi

echo "Dataset builder completed successfully at: $(date)" | tee -a ${LOG_FILE}

# Cleanup is handled by the trap handler
echo "Processing complete. Performing final sync and cleanup..." | tee -a ${LOG_FILE}

# Exit successfully
exit 0
