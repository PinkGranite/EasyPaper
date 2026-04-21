#!/usr/bin/env bash
# Install EasyPaper Claude Code skills and slash commands.
#
# Sources:
#   <repo>/plugins/easypaper/skills/*       -> <target>/skills/<name>/
#   <repo>/plugins/easypaper/commands/*.md  -> <target>/commands/<name>.md
#
# Targets:
#   --global              => $HOME/.claude
#   --project             => $(pwd)/.claude
#   --project=<path>      => <path>/.claude
#
# Modes:
#   --copy (default)      copy files (safe across OSes / no admin needed on Windows)
#   --symlink             create symlinks (updates flow automatically; may need admin on Windows)
#
# Other flags:
#   --list                show what would be installed and exit
#   --dry-run             show actions without writing
#   --uninstall           remove only items this script would have installed
#   -y, --yes             do not prompt for overwrite confirmation
#   -h, --help            show help

set -euo pipefail

PLUGIN_DIR_NAME="easypaper"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLUGIN_ROOT="$REPO_ROOT/plugins/$PLUGIN_DIR_NAME"
SKILLS_SRC="$PLUGIN_ROOT/skills"
COMMANDS_SRC="$PLUGIN_ROOT/commands"

GLOBAL=0
PROJECT=0
PROJECT_PATH=""
MODE="copy"
ACTION="install"
DRY_RUN=0
LIST_ONLY=0
ASSUME_YES=0

usage() {
  cat <<EOF
Install EasyPaper Claude Code skills and slash commands.

Usage:
  $(basename "$SCRIPT_PATH") --global   [options]
  $(basename "$SCRIPT_PATH") --project  [options]
  $(basename "$SCRIPT_PATH") --project=/path/to/project [options]

Targets (one is required, unless --list is used):
  --global              Install to \$HOME/.claude (user-level skills/commands)
  --project             Install to \$(pwd)/.claude
  --project=<path>      Install to <path>/.claude

Modes (mutually exclusive):
  --copy                Copy files (default; safest cross-platform)
  --symlink             Create symlinks (updates from this repo flow automatically;
                        may need admin/developer mode on Windows)

Other:
  --list                Print what would be installed/removed and exit
  --dry-run             Show actions without writing
  --uninstall           Remove only the items this script would install
  -y, --yes             Do not prompt for overwrite confirmation
  -h, --help            Show this help

Examples:
  # First-time global install (user-level)
  bash scripts/install_plugin.sh --global

  # Install into the current repo as a project-scoped plugin
  bash scripts/install_plugin.sh --project

  # Symlink into a specific project for live updates
  bash scripts/install_plugin.sh --project=/home/me/my-research --symlink

  # Just see what would happen
  bash scripts/install_plugin.sh --global --dry-run
  bash scripts/install_plugin.sh --list

  # Clean removal
  bash scripts/install_plugin.sh --global --uninstall -y

Source plugin root: $PLUGIN_ROOT
EOF
}

err() { printf 'error: %s\n' "$*" >&2; }

# ---- arg parsing -----------------------------------------------------------

for arg in "$@"; do
  case "$arg" in
    --global)            GLOBAL=1 ;;
    --project)           PROJECT=1 ;;
    --project=*)         PROJECT=1; PROJECT_PATH="${arg#*=}" ;;
    --copy)              MODE="copy" ;;
    --symlink)           MODE="symlink" ;;
    --list)              LIST_ONLY=1 ;;
    --dry-run)           DRY_RUN=1 ;;
    --uninstall)         ACTION="uninstall" ;;
    -y|--yes)            ASSUME_YES=1 ;;
    -h|--help)           usage; exit 0 ;;
    *)
      err "unknown argument: $arg"
      usage
      exit 2
      ;;
  esac
done

if [[ ! -d "$SKILLS_SRC" ]] || [[ ! -d "$COMMANDS_SRC" ]]; then
  err "plugin source not found:"
  err "  skills:   $SKILLS_SRC"
  err "  commands: $COMMANDS_SRC"
  exit 1
fi

# ---- list-only short-circuit ----------------------------------------------

list_sources() {
  printf 'Plugin source: %s\n\n' "$PLUGIN_ROOT"
  printf 'Skills (%d):\n' "$(find "$SKILLS_SRC" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
  for d in "$SKILLS_SRC"/*/; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    printf '  - %s\n' "$name"
  done
  printf '\nCommands (%d):\n' "$(find "$COMMANDS_SRC" -mindepth 1 -maxdepth 1 -name '*.md' | wc -l | tr -d ' ')"
  for f in "$COMMANDS_SRC"/*.md; do
    [[ -f "$f" ]] || continue
    name="$(basename "$f")"
    printf '  - /%s -> %s\n' "${name%.md}" "$name"
  done
}

if [[ $LIST_ONLY -eq 1 && $GLOBAL -eq 0 && $PROJECT -eq 0 ]]; then
  list_sources
  exit 0
fi

# ---- target resolution ----------------------------------------------------

if [[ $GLOBAL -eq 0 && $PROJECT -eq 0 ]]; then
  err "must specify --global or --project[=<path>]"
  usage
  exit 2
fi

if [[ $GLOBAL -eq 1 && $PROJECT -eq 1 ]]; then
  err "--global and --project are mutually exclusive"
  exit 2
fi

if [[ $GLOBAL -eq 1 ]]; then
  TARGET_ROOT="$HOME/.claude"
  TARGET_LABEL="global ($TARGET_ROOT)"
else
  if [[ -z "$PROJECT_PATH" ]]; then
    PROJECT_PATH="$(pwd)"
  fi
  if [[ ! -d "$PROJECT_PATH" ]]; then
    err "project path does not exist: $PROJECT_PATH"
    exit 1
  fi
  TARGET_ROOT="$(cd "$PROJECT_PATH" && pwd)/.claude"
  TARGET_LABEL="project ($TARGET_ROOT)"
fi

TARGET_SKILLS="$TARGET_ROOT/skills"
TARGET_COMMANDS="$TARGET_ROOT/commands"

# ---- helpers --------------------------------------------------------------

run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '[dry-run] %s\n' "$*"
  else
    eval "$@"
  fi
}

confirm() {
  local prompt="$1"
  if [[ $ASSUME_YES -eq 1 || $DRY_RUN -eq 1 ]]; then
    return 0
  fi
  printf '%s [y/N] ' "$prompt"
  read -r reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

ensure_dir() {
  local d="$1"
  if [[ ! -d "$d" ]]; then
    run "mkdir -p \"$d\""
  fi
}

# Use cp -R for copy; ln -s for symlink. Always remove existing target first.
install_skill() {
  local src_dir="$1" name="$2"
  local dst="$TARGET_SKILLS/$name"
  if [[ -e "$dst" || -L "$dst" ]]; then
    run "rm -rf \"$dst\""
  fi
  if [[ "$MODE" == "symlink" ]]; then
    run "ln -s \"$src_dir\" \"$dst\""
  else
    run "cp -R \"$src_dir\" \"$dst\""
  fi
}

install_command() {
  local src_file="$1" name="$2"
  local dst="$TARGET_COMMANDS/$name"
  if [[ -e "$dst" || -L "$dst" ]]; then
    run "rm -f \"$dst\""
  fi
  if [[ "$MODE" == "symlink" ]]; then
    run "ln -s \"$src_file\" \"$dst\""
  else
    run "cp \"$src_file\" \"$dst\""
  fi
}

remove_item() {
  local path="$1"
  if [[ -e "$path" || -L "$path" ]]; then
    run "rm -rf \"$path\""
  fi
}

# ---- enumerate items ------------------------------------------------------

# mapfile replacement for bash 3.2 compatibility
SKILL_DIRS=()
while IFS= read -r d; do SKILL_DIRS+=("$d"); done < <(find "$SKILLS_SRC" -mindepth 1 -maxdepth 1 -type d | sort)

COMMAND_FILES=()
while IFS= read -r f; do COMMAND_FILES+=("$f"); done < <(find "$COMMANDS_SRC" -mindepth 1 -maxdepth 1 -name '*.md' | sort)

if [[ ${#SKILL_DIRS[@]} -eq 0 && ${#COMMAND_FILES[@]} -eq 0 ]]; then
  err "no skills or commands found under $PLUGIN_ROOT"
  exit 1
fi

# ---- list (with target context) ------------------------------------------

if [[ $LIST_ONLY -eq 1 ]]; then
  printf 'Target: %s\n' "$TARGET_LABEL"
  printf 'Mode:   %s\n\n' "$MODE"
  printf 'Skills (%d):\n' "${#SKILL_DIRS[@]}"
  for d in "${SKILL_DIRS[@]}"; do
    name="$(basename "$d")"
    printf '  %s/skills/%s/\n' "$TARGET_ROOT" "$name"
  done
  printf '\nCommands (%d):\n' "${#COMMAND_FILES[@]}"
  for f in "${COMMAND_FILES[@]}"; do
    name="$(basename "$f")"
    printf '  %s/commands/%s   (slash: /%s)\n' "$TARGET_ROOT" "$name" "${name%.md}"
  done
  exit 0
fi

# ---- pre-action banner ----------------------------------------------------

printf 'EasyPaper plugin %s\n' "$ACTION"
printf '  source : %s\n' "$PLUGIN_ROOT"
printf '  target : %s\n' "$TARGET_LABEL"
printf '  mode   : %s\n' "$MODE"
if [[ $DRY_RUN -eq 1 ]]; then
  printf '  dry-run: yes\n'
fi
printf '  skills : %d\n  commands: %d\n\n' "${#SKILL_DIRS[@]}" "${#COMMAND_FILES[@]}"

# ---- uninstall ------------------------------------------------------------

if [[ "$ACTION" == "uninstall" ]]; then
  if ! confirm "Remove these items from $TARGET_ROOT ?"; then
    err "aborted"
    exit 1
  fi
  for d in "${SKILL_DIRS[@]}"; do
    name="$(basename "$d")"
    remove_item "$TARGET_SKILLS/$name"
  done
  for f in "${COMMAND_FILES[@]}"; do
    name="$(basename "$f")"
    remove_item "$TARGET_COMMANDS/$name"
  done
  printf '\nUninstall complete.\n'
  exit 0
fi

# ---- install --------------------------------------------------------------

# Detect existing items that would be overwritten and warn.
overwrite_count=0
for d in "${SKILL_DIRS[@]}"; do
  name="$(basename "$d")"
  if [[ -e "$TARGET_SKILLS/$name" || -L "$TARGET_SKILLS/$name" ]]; then
    overwrite_count=$((overwrite_count + 1))
  fi
done
for f in "${COMMAND_FILES[@]}"; do
  name="$(basename "$f")"
  if [[ -e "$TARGET_COMMANDS/$name" || -L "$TARGET_COMMANDS/$name" ]]; then
    overwrite_count=$((overwrite_count + 1))
  fi
done

if [[ $overwrite_count -gt 0 ]]; then
  if ! confirm "$overwrite_count existing item(s) at the target will be overwritten. Continue?"; then
    err "aborted"
    exit 1
  fi
fi

ensure_dir "$TARGET_SKILLS"
ensure_dir "$TARGET_COMMANDS"

for d in "${SKILL_DIRS[@]}"; do
  name="$(basename "$d")"
  install_skill "$d" "$name"
  printf '  installed skill   : %s\n' "$name"
done

for f in "${COMMAND_FILES[@]}"; do
  name="$(basename "$f")"
  install_command "$f" "$name"
  printf '  installed command : /%s\n' "${name%.md}"
done

printf '\nDone. Restart Claude Code (or rescan plugins) to pick up changes.\n'
