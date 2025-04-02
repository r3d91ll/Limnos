#!/bin/bash
# Benchmarking script for PathRAG Dataset Builder
# Runs multiple configurations and compares performance

cd "$(dirname "$0")/.."
SCRIPT_DIR="$(pwd)/scripts"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="../logs/benchmarks/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# Default settings
RAG_IMPL="pathrag"
THREADS=24

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --pathrag|--graphrag|--literag)
      RAG_IMPL="${1#--}"
      shift
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--pathrag|--graphrag|--literag] [--threads N]"
      exit 1
      ;;
  esac
done

echo "Running benchmarks for ${RAG_IMPL} implementation with ${THREADS} threads"
echo "Results will be saved to ${LOG_DIR}"

# Run CPU benchmark
echo "Running CPU benchmark..."
${SCRIPT_DIR}/run.sh --cpu --${RAG_IMPL} --threads ${THREADS} 2>&1 | tee "${LOG_DIR}/cpu_benchmark.log"

# Run GPU benchmark
echo "Running GPU benchmark..."
${SCRIPT_DIR}/run.sh --gpu --${RAG_IMPL} 2>&1 | tee "${LOG_DIR}/gpu_benchmark.log"

# Run RAM disk CPU benchmark
echo "Running RAM disk CPU benchmark..."
${SCRIPT_DIR}/run_with_ramdisk.sh --cpu --${RAG_IMPL} --threads ${THREADS} 2>&1 | tee "${LOG_DIR}/ramdisk_cpu_benchmark.log"

# Run RAM disk GPU benchmark
echo "Running RAM disk GPU benchmark..."
${SCRIPT_DIR}/run_with_ramdisk.sh --gpu --${RAG_IMPL} 2>&1 | tee "${LOG_DIR}/ramdisk_gpu_benchmark.log"

# Compare results
echo "Benchmark results:"
echo "================="

# Extract processing times
CPU_TIME=$(grep "Total processing time:" "${LOG_DIR}/cpu_benchmark.log" | awk '{print $4}')
GPU_TIME=$(grep "Total processing time:" "${LOG_DIR}/gpu_benchmark.log" | awk '{print $4}')
RAMDISK_CPU_TIME=$(grep "Total processing time:" "${LOG_DIR}/ramdisk_cpu_benchmark.log" | awk '{print $4}')
RAMDISK_GPU_TIME=$(grep "Total processing time:" "${LOG_DIR}/ramdisk_gpu_benchmark.log" | awk '{print $4}')

# Display times
echo "Standard CPU time: ${CPU_TIME:-N/A} seconds"
echo "Standard GPU time: ${GPU_TIME:-N/A} seconds"
echo "RAM disk CPU time: ${RAMDISK_CPU_TIME:-N/A} seconds"
echo "RAM disk GPU time: ${RAMDISK_GPU_TIME:-N/A} seconds"

# Calculate speedups
if [[ -n "${CPU_TIME}" && -n "${GPU_TIME}" ]]; then
  CPU_GPU_SPEEDUP=$(echo "scale=2; ${CPU_TIME} / ${GPU_TIME}" | bc)
  echo "GPU vs CPU speedup: ${CPU_GPU_SPEEDUP:-N/A}x"
fi

if [[ -n "${CPU_TIME}" && -n "${RAMDISK_CPU_TIME}" ]]; then
  CPU_RAMDISK_SPEEDUP=$(echo "scale=2; ${CPU_TIME} / ${RAMDISK_CPU_TIME}" | bc)
  echo "RAM disk vs standard CPU speedup: ${CPU_RAMDISK_SPEEDUP:-N/A}x"
fi

if [[ -n "${GPU_TIME}" && -n "${RAMDISK_GPU_TIME}" ]]; then
  GPU_RAMDISK_SPEEDUP=$(echo "scale=2; ${GPU_TIME} / ${RAMDISK_GPU_TIME}" | bc)
  echo "RAM disk vs standard GPU speedup: ${GPU_RAMDISK_SPEEDUP:-N/A}x"
fi

if [[ -n "${CPU_TIME}" && -n "${RAMDISK_GPU_TIME}" ]]; then
  TOTAL_SPEEDUP=$(echo "scale=2; ${CPU_TIME} / ${RAMDISK_GPU_TIME}" | bc)
  echo "Total speedup (RAM disk GPU vs standard CPU): ${TOTAL_SPEEDUP:-N/A}x"
fi

echo ""
echo "Benchmark complete. Detailed logs are available in ${LOG_DIR}"
