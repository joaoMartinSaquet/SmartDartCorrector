#!/bin/sh
echo -ne '\033c\033]0;smartDarts\a'
base_path="$(dirname "$(realpath "$0")")"
"$base_path/smartDartEnv2.0.x86_64" "$@"
