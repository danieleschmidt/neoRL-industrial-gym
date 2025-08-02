#!/bin/bash

# Release automation script for neoRL-industrial-gym
# Handles semantic versioning, changelog generation, and deployment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CHANGELOG_FILE="$PROJECT_ROOT/CHANGELOG.md"
VERSION_FILE="$PROJECT_ROOT/pyproject.toml"

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
Release automation script for neoRL-industrial-gym

Usage: $0 [OPTIONS] COMMAND

COMMANDS:
    prepare         Prepare release (update version, changelog)
    build           Build release artifacts
    publish         Publish release to PyPI and container registry
    tag             Create and push git tag
    full            Run complete release process (prepare -> build -> tag -> publish)

OPTIONS:
    -v, --version VERSION   Specify version (semantic version: x.y.z)
    -t, --type TYPE         Version bump type (major|minor|patch)
    --pre-release          Create pre-release version
    --dry-run              Show what would be done without executing
    --skip-tests           Skip test suite (not recommended)
    --skip-security        Skip security scans (not recommended)
    -h, --help             Show this help

EXAMPLES:
    $0 prepare --type patch           # Bump patch version and update changelog
    $0 prepare --version 1.2.3       # Set specific version
    $0 full --type minor             # Complete release with minor version bump
    $0 publish --version 1.2.3       # Publish existing version

ENVIRONMENT VARIABLES:
    PYPI_TOKEN              PyPI API token for publishing
    GITHUB_TOKEN           GitHub token for release creation
    REGISTRY_TOKEN         Container registry token
    CI                     Set in CI environments
EOF
}

# Get current version from pyproject.toml
get_current_version() {
    if [[ -f "$VERSION_FILE" ]]; then
        grep -E '^version = ' "$VERSION_FILE" | sed 's/version = "\(.*\)"/\1/' | tr -d '"'
    else
        echo "0.0.0"
    fi
}

# Bump version based on type
bump_version() {
    local current="$1"
    local type="$2"
    
    # Split version into components
    local major minor patch
    IFS='.' read -r major minor patch <<< "$current"
    
    case "$type" in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
        *)
            log_error "Invalid version type: $type"
            exit 1
            ;;
    esac
    
    echo "$major.$minor.$patch"
}

# Validate semantic version
validate_version() {
    local version="$1"
    
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9\.-]+)?(\+[a-zA-Z0-9\.-]+)?$ ]]; then
        log_error "Invalid semantic version: $version"
        exit 1
    fi
}

# Update version in pyproject.toml
update_version_file() {
    local new_version="$1"
    
    log_info "Updating version in $VERSION_FILE to $new_version"
    
    if [[ -f "$VERSION_FILE" ]]; then
        # Use sed to update version in pyproject.toml
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS sed
            sed -i '' "s/^version = .*/version = \"$new_version\"/" "$VERSION_FILE"
        else
            # GNU sed
            sed -i "s/^version = .*/version = \"$new_version\"/" "$VERSION_FILE"
        fi
    else
        log_error "Version file not found: $VERSION_FILE"
        exit 1
    fi
}

# Generate changelog entry
generate_changelog_entry() {
    local version="$1"
    local date=$(date +%Y-%m-%d)
    
    log_info "Generating changelog entry for version $version"
    
    # Get commits since last tag
    local last_tag
    if last_tag=$(git describe --tags --abbrev=0 2>/dev/null); then
        log_info "Getting changes since last tag: $last_tag"
        local commits=$(git log --oneline "$last_tag"..HEAD)
    else
        log_info "No previous tags found, getting all commits"
        local commits=$(git log --oneline)
    fi
    
    # Create changelog entry
    local changelog_entry="## [$version] - $date\n\n"
    
    # Categorize commits
    local features=""
    local fixes=""
    local breaking=""
    local other=""
    
    while IFS= read -r line; do
        if [[ -z "$line" ]]; then
            continue
        fi
        
        if [[ "$line" =~ ^[a-f0-9]+[[:space:]]+(feat|feature): ]]; then
            features="$features- ${line#* }\n"
        elif [[ "$line" =~ ^[a-f0-9]+[[:space:]]+(fix|bugfix): ]]; then
            fixes="$fixes- ${line#* }\n"
        elif [[ "$line" =~ ^[a-f0-9]+[[:space:]]+BREAKING[[:space:]]*CHANGE: ]]; then
            breaking="$breaking- ${line#* }\n"
        else
            other="$other- ${line#* }\n"
        fi
    done <<< "$commits"
    
    # Build changelog entry
    if [[ -n "$breaking" ]]; then
        changelog_entry="${changelog_entry}### BREAKING CHANGES\n$breaking\n"
    fi
    
    if [[ -n "$features" ]]; then
        changelog_entry="${changelog_entry}### Features\n$features\n"
    fi
    
    if [[ -n "$fixes" ]]; then
        changelog_entry="${changelog_entry}### Bug Fixes\n$fixes\n"
    fi
    
    if [[ -n "$other" ]]; then
        changelog_entry="${changelog_entry}### Other Changes\n$other\n"
    fi
    
    echo -e "$changelog_entry"
}

# Update changelog
update_changelog() {
    local version="$1"
    
    log_info "Updating changelog for version $version"
    
    # Generate entry
    local entry=$(generate_changelog_entry "$version")
    
    # Create changelog if it doesn't exist
    if [[ ! -f "$CHANGELOG_FILE" ]]; then
        cat > "$CHANGELOG_FILE" << EOF
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

EOF
    fi
    
    # Create temporary file with new entry
    local temp_file=$(mktemp)
    
    # Add new entry after the header
    awk '
        BEGIN { entry_added = 0 }
        /^# Changelog/ { print; getline; print; print ""; }
        /^The format is based on/ { print; getline; print; print ""; }
        /^and this project adheres/ { print; print ""; }
        /^## \[/ && !entry_added { 
            print entry
            entry_added = 1
            print
        }
        !/^# Changelog/ && !/^The format is based on/ && !/^and this project adheres/ {
            if (!entry_added && /^$/) {
                print entry
                entry_added = 1
            }
            print
        }
        END {
            if (!entry_added) {
                print entry
            }
        }
    ' entry="$entry" "$CHANGELOG_FILE" > "$temp_file"
    
    mv "$temp_file" "$CHANGELOG_FILE"
    
    log_success "Changelog updated"
}

# Run tests
run_tests() {
    if [[ "${SKIP_TESTS:-false}" == "true" ]]; then
        log_warning "Skipping tests"
        return
    fi
    
    log_info "Running test suite..."
    
    cd "$PROJECT_ROOT"
    
    # Run linting
    make lint || {
        log_error "Linting failed"
        exit 1
    }
    
    # Run type checking
    make type-check || {
        log_error "Type checking failed"
        exit 1
    }
    
    # Run tests
    make test || {
        log_error "Tests failed"
        exit 1
    }
    
    log_success "All tests passed"
}

# Run security scans
run_security_scans() {
    if [[ "${SKIP_SECURITY:-false}" == "true" ]]; then
        log_warning "Skipping security scans"
        return
    fi
    
    log_info "Running security scans..."
    
    cd "$PROJECT_ROOT"
    
    # Safety validation
    make validate-safety || {
        log_error "Safety validation failed"
        exit 1
    }
    
    # Security scan
    make security-scan || {
        log_error "Security scan failed"
        exit 1
    }
    
    log_success "Security scans passed"
}

# Build release artifacts
build_artifacts() {
    local version="$1"
    
    log_info "Building release artifacts for version $version"
    
    cd "$PROJECT_ROOT"
    
    # Clean previous builds
    make clean
    
    # Build Python package
    log_info "Building Python package..."
    python -m build
    
    # Build container images
    log_info "Building container images..."
    "./scripts/build.sh" all --tag "$version" --sbom --security-scan
    
    log_success "Release artifacts built"
}

# Create git tag
create_tag() {
    local version="$1"
    
    log_info "Creating git tag for version $version"
    
    # Create annotated tag
    git tag -a "v$version" -m "Release version $version"
    
    # Push tag
    if [[ "${DRY_RUN:-false}" == "false" ]]; then
        git push origin "v$version"
        log_success "Tag v$version created and pushed"
    else
        log_info "DRY RUN: Would create and push tag v$version"
    fi
}

# Publish to PyPI
publish_pypi() {
    local version="$1"
    
    if [[ -z "${PYPI_TOKEN:-}" ]]; then
        log_warning "PYPI_TOKEN not set, skipping PyPI publication"
        return
    fi
    
    log_info "Publishing version $version to PyPI"
    
    if [[ "${DRY_RUN:-false}" == "false" ]]; then
        python -m twine upload dist/* --username __token__ --password "$PYPI_TOKEN"
        log_success "Published to PyPI"
    else
        log_info "DRY RUN: Would publish to PyPI"
    fi
}

# Publish container images
publish_containers() {
    local version="$1"
    
    if [[ -z "${REGISTRY_TOKEN:-}" ]]; then
        log_warning "REGISTRY_TOKEN not set, skipping container publication"
        return
    fi
    
    log_info "Publishing container images for version $version"
    
    if [[ "${DRY_RUN:-false}" == "false" ]]; then
        # Login to registry
        echo "$REGISTRY_TOKEN" | docker login ghcr.io -u terragon-labs --password-stdin
        
        # Push images
        "./scripts/build.sh" all --tag "$version" --push
        
        log_success "Container images published"
    else
        log_info "DRY RUN: Would publish container images"
    fi
}

# Create GitHub release
create_github_release() {
    local version="$1"
    
    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
        log_warning "GITHUB_TOKEN not set, skipping GitHub release creation"
        return
    fi
    
    log_info "Creating GitHub release for version $version"
    
    # Extract changelog entry for this version
    local release_notes
    release_notes=$(awk "/^## \[$version\]/{flag=1; next} /^## \[/{flag=0} flag" "$CHANGELOG_FILE")
    
    if [[ "${DRY_RUN:-false}" == "false" ]]; then
        # Create release using GitHub CLI
        gh release create "v$version" \
            --title "Release $version" \
            --notes "$release_notes" \
            dist/*
        
        log_success "GitHub release created"
    else
        log_info "DRY RUN: Would create GitHub release"
    fi
}

# Main release functions
prepare_release() {
    local version="$1"
    
    log_info "Preparing release for version $version"
    
    # Update version file
    update_version_file "$version"
    
    # Update changelog
    update_changelog "$version"
    
    # Commit changes
    if [[ "${DRY_RUN:-false}" == "false" ]]; then
        git add "$VERSION_FILE" "$CHANGELOG_FILE"
        git commit -m "chore: prepare release $version"
        log_success "Release preparation committed"
    else
        log_info "DRY RUN: Would commit release preparation"
    fi
}

publish_release() {
    local version="$1"
    
    log_info "Publishing release $version"
    
    # Publish to PyPI
    publish_pypi "$version"
    
    # Publish container images
    publish_containers "$version"
    
    # Create GitHub release
    create_github_release "$version"
    
    log_success "Release $version published"
}

full_release() {
    local version="$1"
    
    log_info "Starting full release process for version $version"
    
    # Run tests and security scans
    run_tests
    run_security_scans
    
    # Prepare release
    prepare_release "$version"
    
    # Build artifacts
    build_artifacts "$version"
    
    # Create tag
    create_tag "$version"
    
    # Publish release
    publish_release "$version"
    
    log_success "Full release process completed for version $version"
}

# Main function
main() {
    local command=""
    local version=""
    local version_type=""
    local pre_release=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--version)
                version="$2"
                shift 2
                ;;
            -t|--type)
                version_type="$2"
                shift 2
                ;;
            --pre-release)
                pre_release=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-security)
                SKIP_SECURITY=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            prepare|build|publish|tag|full)
                command="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate command
    if [[ -z "$command" ]]; then
        log_error "Command required. Use --help for usage information."
        exit 1
    fi
    
    # Determine version
    if [[ -z "$version" ]]; then
        if [[ -n "$version_type" ]]; then
            local current_version=$(get_current_version)
            version=$(bump_version "$current_version" "$version_type")
        else
            log_error "Either --version or --type must be specified"
            exit 1
        fi
    fi
    
    # Validate version
    validate_version "$version"
    
    # Add pre-release suffix if requested
    if [[ "$pre_release" == "true" ]]; then
        version="$version-rc.$(date +%Y%m%d%H%M%S)"
    fi
    
    log_info "Release configuration:"
    log_info "  Command: $command"
    log_info "  Version: $version"
    log_info "  Pre-release: $pre_release"
    log_info "  Dry run: ${DRY_RUN:-false}"
    
    # Execute command
    case "$command" in
        prepare)
            prepare_release "$version"
            ;;
        build)
            build_artifacts "$version"
            ;;
        tag)
            create_tag "$version"
            ;;
        publish)
            publish_release "$version"
            ;;
        full)
            full_release "$version"
            ;;
        *)
            log_error "Unknown command: $command"
            exit 1
            ;;
    esac
    
    log_success "Release command '$command' completed successfully!"
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi