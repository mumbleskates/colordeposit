#!/usr/bin/env bash
mkdir -p /tmp/tspace
sudo mount -t ramfs -o size=4g ramfs /tmp/tspace
sudo chown postgres:postgres /tmp/tspace
