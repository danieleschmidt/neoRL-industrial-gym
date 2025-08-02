#!/bin/bash

# Build automation script for neoRL-industrial-gym
# Supports multi-platform builds, semantic versioning, and SBOM generation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="neorl-industrial-gym"
REGISTRY="${REGISTRY:-ghcr.io/terragon-labs}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Build automation script for neoRL-industrial-gym

Usage: $0 [OPTIONS] TARGET

TARGETS:
    dev         Build development image
    prod        Build production image
    gpu         Build GPU-enabled image
    all         Build all targets
    multi       Build multi-platform images (requires buildx)

OPTIONS:
    -t, --tag TAG           Specify image tag (default: auto-generated)
    -p, --platform PLATFORM Build for specific platform(s)
    --push                  Push to registry after build
    --no-cache             Don't use build cache
    --sbom                 Generate SBOM (Software Bill of Materials)
    --security-scan        Run security scan after build
    --clean                Clean build environment before building
    -v, --verbose          Verbose output
    -h, --help             Show this help

EXAMPLES:
    $0 dev                          # Build development image
    $0 prod --tag v1.0.0 --push     # Build and push production image
    $0 multi --platform linux/amd64,linux/arm64 --push
    $0 all --sbom --security-scan   # Build all with SBOM and security scan

ENVIRONMENT VARIABLES:
    REGISTRY                Container registry (default: ghcr.io/terragon-labs)
    DOCKER_BUILDKIT         Enable BuildKit (recommended: 1)
    CI                      Set in CI environments for appropriate defaults
EOF
}

# Version detection
detect_version() {
    local version=""
    
    # Try to get version from git tag
    if version=$(git describe --tags --exact-match 2>/dev/null); then
        echo "${version#v}"  # Remove 'v' prefix if present
        return
    fi
    
    # Try to get version from pyproject.toml
    if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        version=$(grep -E '^version = ' "$PROJECT_ROOT/pyproject.toml" | sed 's/version = "\(.*\)"/\1/' | tr -d '"')
        if [[ -n "$version" ]]; then
            echo "$version"
            return
        fi
    fi
    
    # Fallback to git describe
    if version=$(git describe --tags --abbrev=7 2>/dev/null); then
        echo "${version#v}"
        return
    fi
    
    # Final fallback
    echo "dev-$(git rev-parse --short HEAD)"
}

# Platform detection
detect_platform() {
    if [[ -n "${PLATFORM:-}" ]]; then
        echo "$PLATFORM"
    elif [[ "$(uname -m)" == "arm64" ]] || [[ "$(uname -m)" == "aarch64" ]]; then
        echo "linux/arm64"
    else
        echo "linux/amd64"
    fi
}

# Check Docker BuildKit
check_buildkit() {
    if [[ "${DOCKER_BUILDKIT:-}" != "1" ]]; then
        log_warning "Docker BuildKit not enabled. Consider setting DOCKER_BUILDKIT=1"
    fi
    
    # Check if buildx is available for multi-platform builds
    if ! docker buildx version >/dev/null 2>&1; then
        log_warning "Docker buildx not available. Multi-platform builds will not work."
        return 1
    fi
    
    return 0
}

# Clean build environment
clean_build() {
    log_info "Cleaning build environment..."
    
    # Remove dangling images
    docker image prune -f >/dev/null 2>&1 || true
    
    # Remove build cache if requested
    if [[ "${NO_CACHE:-false}" == "true" ]]; then
        docker builder prune -f >/dev/null 2>&1 || true
    fi
    
    log_success "Build environment cleaned"
}

# Generate SBOM
generate_sbom() {
    local image="$1"
    local output_file="$PROJECT_ROOT/sbom-${image##*/}.json"
    
    log_info "Generating SBOM for $image..."
    
    # Use syft to generate SBOM if available
    if command -v syft >/dev/null 2>&1; then
        syft "$image" -o json > "$output_file"
        log_success "SBOM generated: $output_file"
    else
        log_warning "syft not found. Installing via Docker..."
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            -v "$PWD":/src \
            anchore/syft:latest \
            "$image" -o json > "$output_file"
        log_success "SBOM generated: $output_file"
    fi
}

# Security scan
security_scan() {
    local image="$1"
    
    log_info "Running security scan for $image..."
    
    # Use grype for vulnerability scanning if available
    if command -v grype >/dev/null 2>&1; then
        grype "$image" --output table
    else
        log_warning "grype not found. Using Docker for scanning..."
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            anchore/grype:latest \
            "$image" --output table
    fi
}

# Build single target
build_target() {
    local target="$1"
    local tag="$2"
    local platform="$3"
    
    local image_ref="$REGISTRY/$IMAGE_NAME:$tag-$target"
    local latest_ref="$REGISTRY/$IMAGE_NAME:$target"
    
    log_info "Building $target image: $image_ref"
    log_info "Platform: $platform"
    
    # Build arguments
    local build_args=(
        --build-arg BUILD_DATE="$BUILD_DATE"
        --build-arg VCS_REF="$VCS_REF"
        --build-arg VERSION="$tag"
        --target "$target"
        --platform "$platform"
        --tag "$image_ref"
        --tag "$latest_ref"
    )
    
    # Add cache options
    if [[ "${NO_CACHE:-false}" == "true" ]]; then
        build_args+=(--no-cache)
    fi
    
    # Add push option
    if [[ "${PUSH:-false}" == "true" ]]; then
        build_args+=(--push)
    else
        build_args+=(--load)
    fi
    
    # Use buildx for multi-platform or regular docker build
    if [[ "$platform" == *","* ]] || [[ "${PUSH:-false}" == "true" ]]; then
        # Multi-platform or push requires buildx
        docker buildx build "${build_args[@]}" "$PROJECT_ROOT"
    else
        # Single platform, can use regular build
        docker build "${build_args[@]}" "$PROJECT_ROOT"
    fi
    
    # Post-build actions (only for single platform, local builds)
    if [[ "$platform" != *","* ]] && [[ "${PUSH:-false}" == "false" ]]; then
        # Generate SBOM if requested
        if [[ "${GENERATE_SBOM:-false}" == "true" ]]; then
            generate_sbom "$image_ref"
        fi
        
        # Security scan if requested
        if [[ "${SECURITY_SCAN:-false}" == "true" ]]; then
            security_scan "$image_ref"
        fi
    fi
    
    log_success "Successfully built $target: $image_ref"
}

# Main build function
main() {
    local target=""
    local tag=""
    local platform=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--tag)
                tag="$2"
                shift 2
                ;;
            -p|--platform)
                platform="$2"
                shift 2
                ;;
            --push)
                PUSH=true
                shift
                ;;
            --no-cache)
                NO_CACHE=true
                shift
                ;;
            --sbom)
                GENERATE_SBOM=true
                shift
                ;;
            --security-scan)
                SECURITY_SCAN=true
                shift
                ;;
            --clean)
                CLEAN=true
                shift
                ;;
            -v|--verbose)
                set -x
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            dev|prod|gpu|all|multi)
                target="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate target
    if [[ -z "$target" ]]; then
        log_error "Target required. Use --help for usage information."
        exit 1
    fi
    
    # Detect version and platform if not specified
    tag="${tag:-$(detect_version)}"
    platform="${platform:-$(detect_platform)}"
    
    log_info "Build configuration:"
    log_info "  Target: $target"
    log_info "  Tag: $tag"
    log_info "  Platform: $platform"
    log_info "  Registry: $REGISTRY"
    log_info "  Push: ${PUSH:-false}"
    log_info "  Generate SBOM: ${GENERATE_SBOM:-false}"
    log_info "  Security Scan: ${SECURITY_SCAN:-false}"
    
    # Check prerequisites
    check_buildkit || {
        if [[ "$target" == "multi" ]] || [[ "$platform" == *","* ]]; then
            log_error "Multi-platform builds require Docker buildx"
            exit 1
        fi
    }
    
    # Clean if requested
    if [[ "${CLEAN:-false}" == "true" ]]; then
        clean_build
    fi
    
    # Build targets
    case "$target" in
        dev)
            build_target "development" "$tag" "$platform"
            ;;
        prod)
            build_target "production" "$tag" "$platform"
            ;;
        gpu)
            build_target "gpu" "$tag" "$platform"
            ;;
        all)
            build_target "development" "$tag" "$platform"
            build_target "production" "$tag" "$platform"
            build_target "gpu" "$tag" "$platform"
            ;;
        multi)
            # Multi-platform builds
            local multi_platform="${platform:-linux/amd64,linux/arm64}"
            log_info "Building multi-platform images for: $multi_platform"
            
            build_target "development" "$tag" "$multi_platform"
            build_target "production" "$tag" "$multi_platform"
            # GPU images are typically amd64 only due to NVIDIA CUDA
            build_target "gpu" "$tag" "linux/amd64"
            ;;
        *)
            log_error "Unknown target: $target"
            exit 1
            ;;
    esac
    
    log_success "Build completed successfully!"
    
    # Show image information
    if [[ "${PUSH:-false}" == "false" ]] && [[ "$platform" != *","* ]]; then
        log_info "Built images:"
        docker images "$REGISTRY/$IMAGE_NAME" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    fi
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi