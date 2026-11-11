#!/bin/zsh
set -euo pipefail

repo_root="${0:A:h:h}"
label="com.aoayhan.basketballdna-live-season"
agents_dir="${HOME}/Library/LaunchAgents"
logs_dir="${HOME}/Library/Logs/BasketballDNA"
plist="${agents_dir}/${label}.plist"
mkdir -p "${agents_dir}" "${logs_dir}"

cat > "${plist}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${label}</string>
  <key>ProgramArguments</key>
  <array><string>${repo_root}/scripts/update_and_publish_live_season.sh</string></array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>10</integer><key>Minute</key><integer>0</integer></dict>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>900</integer>
  <key>StandardOutPath</key><string>${logs_dir}/live-season.log</string>
  <key>StandardErrorPath</key><string>${logs_dir}/live-season-error.log</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/${UID}" "${plist}" 2>/dev/null || true
launchctl bootstrap "gui/${UID}" "${plist}"
echo "Installed ${label}; it will run daily at 10:00 local time and retry failed runs after 15 minutes."
