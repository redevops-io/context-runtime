# docker-contextos — an AppArmor profile for the LibreQB × FinanceBench stack.
#
# It is Docker's stock `docker-default` profile (same /proc, /sys, mount deny
# rules — full confinement), pinned to the AppArmor 3.0 policy ABI.
#
# Why: this host runs apparmor_parser 4.1, which compiles against a newer feature
# ABI whose fine-grained network mediation does NOT let a classic `network,` rule
# cover AF_UNIX socket *creation* on this kernel — every unix-socket create fails
# "protocol match" (audit class="net", family="unix"). That breaks Python/uvicorn's
# socket.socketpair() (PermissionError [Errno 13]) and mongod's listener socket
# ("open: Permission denied" before WiredTiger). Node/Caddy don't use AF_UNIX at
# startup, which is why only the Python + Mongo services crashed under docker-default.
#
# Pinning `abi <abi/3.0>,` compiles the same rules under the 3.0 ABI, where
# `network,` mediates AF_UNIX creation the classic (working) way — full confinement
# kept, no `unconfined` needed. `unix,` additionally covers peer connect/send ops.
#
# Load:  apparmor_parser -r -W /etc/apparmor.d/docker-contextos
# Use:   security_opt: ["apparmor=docker-contextos"]  (see chat-demo.compose.yml)

abi <abi/3.0>,
#include <tunables/global>

profile docker-contextos flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>

  network,
  unix,
  capability,
  file,
  umount,

  # Host (privileged) processes may send signals to container processes.
  signal (receive) peer=unconfined,
  # dockerd may send signals to container processes.
  signal (receive) peer=dockerd,
  # Container processes may send signals amongst themselves.
  signal (send,receive) peer=docker-contextos,

  deny @{PROC}/* w,   # deny write for all files directly in /proc (not in a subdir)
  # deny write to files not in /proc/<number>/** or /proc/sys/**
  deny @{PROC}/{[^1-9],[^1-9][^0-9],[^1-9s][^0-9y][^0-9s],[^1-9][^0-9][^0-9][^0-9]*}/** w,
  deny @{PROC}/sys/[^k]** w,  # deny /proc/sys except /proc/sys/k* (effectively /proc/sys/kernel)
  deny @{PROC}/sys/kernel/{?,??,[^s][^h][^m]**} w,  # deny everything except shm* properties
  deny @{PROC}/sysrq-trigger rwklx,
  deny @{PROC}/kcore rwklx,

  deny mount,

  deny /sys/[^f]*/** wklx,
  deny /sys/f[^s]*/** wklx,
  deny /sys/fs/[^c]*/** wklx,
  deny /sys/fs/c[^g]*/** wklx,
  deny /sys/fs/cg[^r]*/** wklx,
  deny /sys/firmware/** rwklx,
  deny /sys/kernel/security/** rwklx,
}
